from typing import List, Dict, Optional, Any, TypedDict, Protocol
from datetime import datetime
import json
import os
import asyncio
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
from scripts.models import (
    KnowledgeAcquisitionConfig,
    ExtractedKnowledge,
    Relationship,
    SourceMetadata,
    Entity
)
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
            temperature=0.7,
            mirostat=2,
            mirostat_eta=0.1,
            mirostat_tau=5.0
        )
        super().__init__(llm)
        
        # Initialize state
        self.state = ProcessingState()
        
        # Initialize other components
        self.config = config
        self.embeddings = None
        self.vector_store = None
        self.web_browser = None
        self.text_inspector = None
        self.visual_analyzer = None
        self.graph = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize parsers
        self.entity_parser = PydanticOutputParser(pydantic_object=EntityResponse)
        self.relationship_parser = PydanticOutputParser(pydantic_object=RelationshipResponse)
        self.metadata_parser = PydanticOutputParser(pydantic_object=MetadataResponse)
        
        log_info_with_context("Knowledge acquisition system initialized", "Initialization")

    async def _generate_plan(self, state: CompilerState) -> Plan:
        """Generate knowledge acquisition plan"""
        try:
            prompt = get_plan_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Plan)
            plan = await chain.ainvoke({
                "content": state.content
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
            # Format state for LLM using direct attribute access
            formatted_state = {
                "plan": state.plan.model_dump() if state.plan else None,
                "results": [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in state.results],
                "current_progress": {
                    "knowledge_sources": len(self.state.tasks),
                    "synthetic_knowledge": len(self.state.completed),
                    "training_examples": len(self.state.errors),
                    "has_metrics": bool(self.state.errors)
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

    async def _generate_final_result(self, state: CompilerState) -> List[Document]:
        """Generate final documents from results"""
        try:
            documents = []
            for result in state.results:  # Access results directly
                if result and result.result and isinstance(result.result, Document):
                    documents.append(result.result)
            return documents

        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise

    async def add_source(self, source_path: str, source_type: str) -> List[Document]:
        """Add a source to the knowledge base"""
        try:
            log_info_with_context(f"Processing source: {source_path}", "Knowledge Processing")
            
            # Load source content
            docs = []
            if source_type == "text":
                loader = TextLoader(source_path)
                docs = await asyncio.to_thread(loader.load)
            elif source_type == "pdf":
                loader = PyPDFLoader(source_path)
                docs = await asyncio.to_thread(loader.load)
            elif source_type == "web":
                # Use web browser to fetch content
                content = await self.web_browser.get_content(source_path)
                if content:
                    docs = [Document(
                        page_content=content,
                        metadata={"source": source_path, "type": "web"}
                    )]
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
            if not docs:
                log_warning_with_context(f"No content extracted from source: {source_path}", "Knowledge Processing")
                return []
                
            # Split documents
            split_docs = await asyncio.to_thread(
                self.text_splitter.split_documents,
                docs
            )
            
            # Generate embeddings and add to vector store
            if self.vector_store:
                try:
                    await asyncio.to_thread(
                        self.vector_store.add_documents,
                        split_docs
                    )
                    log_info_with_context(f"Added {len(split_docs)} documents to vector store", "Knowledge Processing")
                except Exception as e:
                    log_error_with_traceback(e, "Error adding documents to vector store")
                    
            # Process each chunk
            processed_docs = []
            for doc in split_docs:
                try:
                    # Extract knowledge
                    knowledge = await self.process_source(doc.page_content)
                    if knowledge and knowledge.confidence >= self.config.confidence_thresholds["extraction"]:
                        # Add metadata
                        doc.metadata.update({
                            "knowledge_extracted": True,
                            "confidence": knowledge.confidence,
                            "entities": [e.name for e in knowledge.entities],
                            "relationships": [r.model_dump() for r in knowledge.relationships],
                            "timestamp": datetime.now().isoformat()
                        })
                        processed_docs.append(doc)
                        
                        # Add to graph database
                        if self.graph:
                            try:
                                await self._add_to_graph(knowledge)
                                log_info_with_context("Added knowledge to graph database", "Knowledge Processing")
                            except Exception as e:
                                log_error_with_traceback(e, "Error adding to graph database")
                                
                except Exception as e:
                    log_error_with_traceback(e, f"Error processing document chunk: {e}")
                    continue
                    
            log_info_with_context(f"Successfully processed {len(processed_docs)} documents", "Knowledge Processing")
            return processed_docs
            
        except Exception as e:
            log_error_with_traceback(e, "Error adding source")
            raise

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
            log_warning_with_context("LLM not initialized, initializing now", "Knowledge Processing")
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
                
                try:
                    # Run text analysis first
                    log_info_with_context("Starting text analysis", "Knowledge Processing")
                    inspector = TextInspector(self.llm)
                    analysis: TextAnalysis = await inspector.inspect_text(content)
                    progress.update(extraction_task, advance=1)
                    
                    # Extract knowledge using knowledge extraction prompt
                    log_info_with_context("Extracting initial knowledge", "Knowledge Processing")
                    prompt = get_knowledge_extraction_prompt()
                    chain = prompt | self.llm | PydanticOutputParser(pydantic_object=ExtractedKnowledge)
                    initial_knowledge = await chain.ainvoke({
                        "text": content,
                        "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else analysis
                    })
                    progress.update(extraction_task, advance=1)
                    
                    # Run entity extraction, relationship extraction, and metadata generation in parallel
                    log_info_with_context("Starting parallel knowledge extraction", "Knowledge Processing")
                    tasks = [
                        self._extract_entities(content),
                        self._extract_relationships(content),
                        self._generate_metadata(content)
                    ]
                    
                    entities, relationships, metadata = await asyncio.gather(*tasks)
                    progress.update(extraction_task, advance=1)

                    # Evaluate confidence with factors
                    log_info_with_context("Evaluating confidence", "Knowledge Processing")
                    confidence_eval = await self._evaluate_confidence(content, entities, relationships)
                    factors = confidence_eval.factors
                    
                    # Optionally enrich with web search
                    web_context = None
                    if self.config.enable_web_search:
                        try:
                            log_info_with_context("Enriching with web search", "Knowledge Processing")
                            search_results = await web_search(content)
                            if search_results and not search_results.startswith("Error"):
                                web_context = search_results
                                log_info_with_context("Web search enrichment successful", "Knowledge Processing")
                            else:
                                log_warning_with_context("Web search returned no results", "Knowledge Processing")
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
                    log_info_with_context(
                        f"Knowledge extraction complete - Confidence: {knowledge.confidence:.2f}",
                        "Knowledge Processing"
                    )
                    return knowledge
                    
                except Exception as e:
                    log_error_with_traceback(e, "Error in knowledge extraction process")
                    raise
                
        except Exception as e:
            log_error_with_traceback(e, "Fatal error in process_source")
            raise

    async def _extract_entities(self, content: str) -> List[Entity]:
        """Extract entities from content"""
        try:
            log_info_with_context("Starting entity extraction", "Knowledge Processing")
            if not self.llm:
                raise ValueError("LLM not initialized")
            
            prompt = get_entity_extraction_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=EntityResponse)
            result = await chain.ainvoke({
                "content": content
            })
            
            # Convert string entities to Entity objects
            entities = []
            for entity_str in result.entities:
                entities.append(Entity(
                    name=entity_str,
                    type="unknown",
                    confidence=0.7
                ))
            
            return entities
            
        except Exception as e:
            log_error_with_traceback(e, "Error extracting entities")
            raise

    async def _extract_relationships(self, content: str) -> List[Relationship]:
        """Extract relationships from content"""
        try:
            log_info_with_context("Starting relationship extraction", "Knowledge Processing")
            await self._ensure_initialized()
            if not self.llm:
                raise ValueError("LLM not initialized")
            
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
                
            log_info_with_context(f"Extracted {len(relationships)} relationships", "Knowledge Processing")
            return relationships
            
        except Exception as e:
            log_error_with_traceback(e, "Error extracting relationships")
            raise

    async def _generate_metadata(self, content: str) -> SourceMetadata:
        """Generate metadata for content"""
        try:
            log_info_with_context("Starting metadata generation", "Knowledge Processing")
            await self._ensure_initialized()
            if not self.llm:
                raise ValueError("LLM not initialized")
            
            prompt = get_metadata_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=MetadataResponse)
            result = await chain.ainvoke({
                "content": content
            })
            
            log_info_with_context("Metadata generation complete", "Knowledge Processing")
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
        """Initialize the system"""
        try:
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model=os.getenv("OLLAMA_MODEL", "MFDoom/deepseek-r1-tool-calling:1.5b")
            )
            
            # Initialize vector store
            self.vector_store = Chroma(
                collection_name="knowledge_store",
                embedding_function=self.embeddings,
                persist_directory="./data/vector_store"
            )
            
            # Initialize web browser for crawling
            self.web_browser = SimpleTextBrowser()
            
            # Initialize text inspector
            self.text_inspector = TextInspector()
            
            # Initialize visual analyzer
            self.visual_analyzer = VisualAnalyzer()
            
            # Initialize Neo4j connection
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
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

    async def _generate_embeddings(self, content: str) -> Optional[List[float]]:
        """Generate embeddings with proper error handling"""
        try:
            log_info_with_context("Starting embeddings generation", "Embeddings")
            if not self.embeddings:
                log_warning_with_context("Embeddings not initialized, initializing now", "Embeddings")
                await self.initialize()
                if not self.embeddings:
                    raise ValueError("Failed to initialize embeddings")
            
            # Use a safe wrapper to handle the None case
            def generate_embeddings(text: str) -> Optional[List[List[float]]]:
                if self.embeddings:
                    try:
                        return self.embeddings.embed_documents([text])
                    except Exception as e:
                        log_error_with_traceback(e, "Error in embed_documents call")
                        return None
                return None
                
            embeddings = await asyncio.to_thread(generate_embeddings, content)
            if embeddings and isinstance(embeddings, list):
                log_info_with_context(f"Successfully generated embeddings of length {len(embeddings[0])}", "Embeddings")
                return embeddings[0]
            
            log_warning_with_context("Failed to generate embeddings - returned None or empty list", "Embeddings")
            return None
            
        except Exception as e:
            log_error_with_traceback(e, "Fatal error generating embeddings")
            raise