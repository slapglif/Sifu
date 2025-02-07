from typing import List, Dict, Optional, Any, Literal, cast, Union, TypedDict, Mapping, Protocol
from datetime import datetime
import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_neo4j import Neo4jGraph
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
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
from scripts.visual_qa import visualizer
from scripts.text_inspector_tool import TextInspector
import os
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph

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
            result = await asyncio.to_thread(
                self.inspector.inspect_text,
                content,
                kwargs.get("metadata", {})
            )
            return result.dict()
        except Exception as e:
            log_error_with_traceback(e, "Error in text inspection")
            return {}

class VisualProcessor(ContentProcessor):
    """Visual processor using LLMCompiler pattern"""
    async def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """Process visual content"""
        try:
            if "image_path" not in kwargs:
                return {}
                
            result = await visualizer(
                kwargs["image_path"],
                kwargs.get("question", "Describe this image in detail.")
            )
            return json.loads(result)
        except Exception as e:
            log_error_with_traceback(e, "Error in visual processing")
            return {}

class DocumentProcessor:
    """Document processor using LLMCompiler pattern"""
    def __init__(self, llm):
        self.markdown_converter = MarkdownConverter()
        self.text_processor = TextInspectorProcessor(llm)
        self.visual_processor = VisualProcessor()
        
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

class EntityResponse(BaseModel):
    """Schema for entity extraction response"""
    entities: List[str] = Field(description="List of extracted entities")

class RelationshipResponse(BaseModel):
    """Schema for relationship extraction response"""
    relationships: List[Relationship] = Field(description="List of extracted relationships")

class MetadataResponse(BaseModel):
    """Schema for metadata generation response"""
    metadata: SourceMetadata = Field(description="Generated metadata")

class DocumentChunk(BaseModel):
    """Schema for document chunks"""
    content: str = Field(description="Chunk content")
    metadata: Dict = Field(description="Chunk metadata")

class DocumentResponse(BaseModel):
    """Schema for document loading response"""
    chunks: List[DocumentChunk] = Field(description="List of document chunks")
    source_path: str = Field(description="Source file path")
    source_type: str = Field(description="Source type")

class TaskState(BaseModel):
    """State for task execution"""
    content: str = Field(description="Content to process")
    knowledge: Optional[ExtractedKnowledge] = None
    embeddings: Optional[List[float]] = None
    graph_updates: Optional[bool] = Field(default=None, description="Whether graph updates are complete")
    error: Optional[str] = None

class ProcessingResult(TypedDict):
    """Result of processing"""
    tasks: List[Dict[str, Any]]
    completed: List[Document]
    errors: List[str]

class ProcessingState(BaseModel):
    """Overall processing state"""
    tasks: List[TaskState] = Field(default_factory=list)
    completed: List[Document] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def model_dump(self) -> Dict[str, Any]:
        """Convert state to dict"""
        return {
            "tasks": [task.model_dump() for task in self.tasks],
            "completed": [doc.dict() for doc in self.completed],
            "errors": [err for err in self.errors]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingState":
        """Create state from dict"""
        return cls(
            tasks=[TaskState(**task) for task in data.get("tasks", [])],
            completed=[Document(**doc) for doc in data.get("completed", [])],
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

class KnowledgeAcquisitionSystem:
    """Knowledge acquisition system with LLMCompiler pattern"""
    def __init__(self, config: KnowledgeAcquisitionConfig):
        """Initialize basic attributes"""
        self.logger = logger
        self.config = config
        self.llm = None
        self.embeddings: Embeddings = OllamaEmbeddings(model='bge-m3', base_url='http://localhost:11434')
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
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create processing workflow DAG"""
        workflow = StateGraph(ProcessingState)

        # Knowledge extraction node
        async def extract_knowledge(state: ProcessingState) -> ProcessingState:
            """Extract knowledge from content"""
            if not self.llm:
                await self.initialize()

            tasks = []
            for task in state.tasks:
                if task.knowledge is None and not task.error:
                    try:
                        knowledge = await self.process_source(task.content)
                        task.knowledge = knowledge
                    except Exception as e:
                        task.error = str(e)
                tasks.append(task)
            
            state.tasks = tasks
            return state

        # Embedding generation node
        async def generate_embeddings(state: ProcessingState) -> ProcessingState:
            """Generate embeddings in parallel"""
            if not self.embeddings:
                await self.initialize()

            tasks = []
            with ThreadPoolExecutor() as executor:
                futures = []
                for task in state.tasks:
                    if task.knowledge and not task.embeddings and not task.error:
                        futures.append(
                            executor.submit(
                                self.embeddings.embed_documents,
                                [task.knowledge.content]
                            )
                        )
                    else:
                        futures.append(None)

                for task, future in zip(state.tasks, futures):
                    if future:
                        try:
                            embeddings = future.result()
                            task.embeddings = embeddings[0] if isinstance(embeddings, list) else embeddings
                        except Exception as e:
                            task.error = str(e)
                    tasks.append(task)

            state.tasks = tasks
            return state

        # Graph update node
        async def update_graph(state: ProcessingState) -> ProcessingState:
            """Update graph database"""
            if not self.graph:
                await self.initialize()

            tasks = []
            for task in state.tasks:
                if task.knowledge and not task.graph_updates and not task.error:
                    try:
                        await self._add_to_graph(task.knowledge)
                        task.graph_updates = True
                    except Exception as e:
                        task.error = str(e)
                tasks.append(task)

            state.tasks = tasks
            return state

        # Document creation node
        async def create_documents(state: ProcessingState) -> ProcessingState:
            """Create final documents"""
            completed = []
            for task in state.tasks:
                if task.error:
                    state.errors.append(task.error)
                    continue

                if task.knowledge and task.embeddings:
                    doc = Document(
                        page_content=task.knowledge.content,
                        metadata={
                            "entities": task.knowledge.entities,
                            "relationships": [r.model_dump() for r in task.knowledge.relationships],
                            "confidence": task.knowledge.confidence,
                            "domain": task.knowledge.domain
                        }
                    )
                    completed.append(doc)

            state.completed = completed
            return state

        # Add nodes
        workflow.add_node("extract_knowledge", extract_knowledge)
        workflow.add_node("generate_embeddings", generate_embeddings)
        workflow.add_node("update_graph", update_graph)
        workflow.add_node("create_documents", create_documents)

        # Add edges
        workflow.add_edge("extract_knowledge", "generate_embeddings")
        workflow.add_edge("generate_embeddings", "update_graph")
        workflow.add_edge("update_graph", "create_documents")

        workflow.set_entry_point("extract_knowledge")
        workflow.set_finish_point("create_documents")

        return workflow

    async def initialize(self) -> "KnowledgeAcquisitionSystem":
        """Async initialization of components"""
        try:
            # Initialize LLM with strict JSON formatting
            self.llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "MFDoom/deepseek-r1-tool-calling:1.5b"),
                format="json",
                temperature=0.7
            )
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
            
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
            
            self.logger.info("Initialization: Knowledge acquisition system initialized successfully")
            return self
            
        except Exception as e:
            log_error_with_traceback(e, "Error initializing knowledge acquisition system")
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
                            "timestamp": datetime.now().isoformat(),
                            "domain": knowledge.domain
                        }
                    }
                )
            
            # Create relationships
            for rel in knowledge.relationships:
                await asyncio.to_thread(
                    self.graph.query,
                    """
                    MATCH (s:Entity {name: $source}), (t:Entity {name: $target})
                    MERGE (s)-[r:$relation_type {type: $relation}]->(t)
                    SET r += $properties
                    RETURN r
                    """,
                    {
                        "source": rel.source,
                        "target": rel.target,
                        "relation": rel.relation,
                        "relation_type": rel.relation.upper(),
                        "properties": {
                            "confidence": rel.confidence,
                            "domain": rel.domain,
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
                extraction_task = progress.add_task("[cyan]Extracting knowledge...", total=3)
                
                # Run entity extraction, relationship extraction, and metadata generation in parallel
                tasks = [
                    self._extract_entities(content),
                    self._extract_relationships(content),
                    self._generate_metadata(content)
                ]
                
                log_info_with_context("Starting parallel knowledge extraction", "Knowledge Processing")
                entities, relationships, metadata = await asyncio.gather(*tasks)
                progress.update(extraction_task, advance=3)

                # Create knowledge object
                knowledge = ExtractedKnowledge(
                    content=content,
                    entities=entities,
                    relationships=relationships,
                    confidence=metadata.confidence_score,
                    metadata=metadata,
                    domain="knowledge"
                )
                
                # Log extraction results
                log_extraction_results(knowledge)
                return knowledge
                
        except Exception as e:
            log_error_with_traceback(e, "Error in process_source")
            raise

    async def add_source(self, source_path: str, source_type: str) -> List[Document]:
        """Add a knowledge source and extract knowledge from it."""
        try:
            # Get shared progress instance
            progress = create_progress()
            
            # Create progress tasks
            load_task = progress.add_task("[cyan]Loading source...", total=1)
            process_task = progress.add_task("[cyan]Processing chunks...", total=None)
            
            # Handle web sources
            if source_type == "web":
                log_info_with_context(f"Fetching web content from {source_path}", "Source Loading")
                doc = await self._fetch_web_content(source_path)
                if not doc:
                    return []
                chunks = self.text_splitter.split_text(doc.page_content)
            else:
                # Load and split source
                log_info_with_context(f"Loading source file: {source_path}", "Source Loading")
                chunks = await self._load_chunks(source_path, source_type)
            
            progress.update(load_task, advance=1)
            
            if not chunks:
                log_warning_with_context("No content chunks extracted", "Source Loading")
                return []

            # Update progress for processing
            progress.update(process_task, total=len(chunks))
            
            # Create initial state
            state = ProcessingState(tasks=[
                TaskState(content=chunk)
                for chunk in chunks
            ])

            # Run workflow
            app = self.workflow.compile()
            result = await app.ainvoke(state.model_dump())
            final_state = ProcessingState.from_dict(result)
            
            # Log processing stats
            stats_table = Table(title="Processing Results")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            stats_table.add_row("Total Chunks", str(len(chunks)))
            stats_table.add_row("Processed Documents", str(len(final_state.completed)))
            stats_table.add_row("Errors", str(len(final_state.errors)))
            
            console.print(stats_table)
            
            if final_state.errors:
                console.print(Panel(
                    "\n".join(final_state.errors),
                    title="[red]Processing Errors[/red]",
                    border_style="red"
                ))

            return final_state.completed

        except Exception as e:
            log_error_with_traceback(e, f"Error adding source {source_path}")
            return []

    async def search(self, query: str, min_confidence: float = 0.7) -> List[Document]:
        """Combined vector, graph and web search"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                search_task = progress.add_task("[cyan]Searching knowledge sources...", total=3)
                
                # Run searches in parallel
                log_info_with_context(f"Starting combined search for: {query}", "Search")
                vector_task = self._vector_search(query, min_confidence)
                graph_task = self._graph_search(query)
                web_task = self._web_search(query)
                
                vector_results, graph_results, web_results = await asyncio.gather(
                    vector_task, graph_task, web_task
                )
                progress.update(search_task, advance=3)
                
                # Log search stats
                stats_table = Table(title="Search Results")
                stats_table.add_column("Source", style="cyan")
                stats_table.add_column("Results", style="green")
                
                stats_table.add_row("Vector Store", str(len(vector_results)))
                stats_table.add_row("Knowledge Graph", str(len(graph_results)))
                stats_table.add_row("Web Search", str(len(web_results)))
                
                console.print(stats_table)
                
                # Combine and deduplicate results
                all_docs = vector_results + graph_results + web_results
                seen = set()
                unique_docs = []
                
                for doc in all_docs:
                    content = doc.page_content
                    if content not in seen:
                        seen.add(content)
                        unique_docs.append(doc)
                
                log_info_with_context(
                    f"Found {len(unique_docs)} unique results from {len(all_docs)} total",
                    "Search"
                )
                
                return unique_docs
                
        except Exception as e:
            log_error_with_traceback(e, "Error in combined search")
            return []

    async def _vector_search(self, query: str, min_confidence: float) -> List[Document]:
        """Search vector store with proper error handling"""
        await self._ensure_initialized()
        if not self.vector_store:
            return []
            
        try:
            # Convert filter to string-based format expected by Chroma
            filter_dict = {"confidence": str(min_confidence)}
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query,
                k=5,
                filter=filter_dict
            )
            
            return [doc for doc, score in results if score >= min_confidence]
            
        except Exception as e:
            log_error_with_traceback(e, "Error searching vector store")
            return []

    async def _graph_search(self, query: str) -> List[Document]:
        """Search graph database"""
        try:
            if not self.graph:
                return []
                
            # Query graph for relevant nodes and relationships
            results = await asyncio.to_thread(
                self.graph.query,
                """
                CALL db.index.fulltext.queryNodes("entity_content", $query) YIELD node, score
                WHERE score > 0.7
                WITH node
                MATCH (node)-[r]-(related)
                RETURN node, collect(r), collect(related)
                LIMIT 5
                """,
                {"query": query}
            )
            
            # Convert results to documents
            documents = []
            for result in results:
                node = result["node"]
                relationships = result["collect(r)"]
                related_nodes = result["collect(related)"]
                
                doc = Document(
                    page_content=node["content"],
                    metadata={
                        "entities": [node["name"]],
                        "relationships": [
                            {
                                "source": r["source"],
                                "relation": r["type"],
                                "target": r["target"]
                            }
                            for r in relationships
                        ],
                        "confidence": node.get("confidence", 0.0)
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            log_error_with_traceback(e, "Error searching graph")
            return []

    async def _web_search(self, query: str) -> List[Document]:
        """Search web using SearxNG and process results"""
        try:
            log_info_with_context(f"Starting web search for: {query}", "Web Search")
            
            # Perform web search
            search_results = await web_search(query)
            if not search_results or "Error:" in search_results:
                log_warning_with_context("No web search results found", "Web Search")
                return []
            
            # Parse results
            documents = []
            current_doc = {"url": "", "title": "", "content": ""}
            
            for line in search_results.split("\n"):
                if line.startswith("URL: "):
                    if current_doc["url"]:  # Save previous doc
                        doc = Document(
                            page_content=current_doc["content"],
                            metadata={
                                "url": current_doc["url"],
                                "title": current_doc["title"],
                                "source_type": "web",
                                "confidence": 0.7,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                    current_doc = {"url": line[5:], "title": "", "content": ""}
                elif line.startswith("Title: "):
                    current_doc["title"] = line[7:]
                elif line.startswith("Summary: "):
                    current_doc["content"] = line[9:]
            
            # Add last document
            if current_doc["url"]:
                doc = Document(
                    page_content=current_doc["content"],
                    metadata={
                        "url": current_doc["url"],
                        "title": current_doc["title"],
                        "source_type": "web",
                        "confidence": 0.7,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            log_info_with_context(f"Found {len(documents)} web documents", "Web Search")
            return documents
            
        except Exception as e:
            log_error_with_traceback(e, "Error in web search")
            return []

    async def _fetch_web_content(self, url: str) -> Optional[Document]:
        """Fetch and process web content"""
        try:
            log_info_with_context(f"Fetching content from: {url}", "Web Content")
            
            browser = SimpleTextBrowser()
            content = await browser.visit(url)
            
            if not content or "Error visiting page:" in content:
                log_warning_with_context(f"Failed to fetch content from {url}", "Web Content")
                return None
            
            doc = Document(
                page_content=content,
                metadata={
                    "url": url,
                    "source_type": "web",
                    "confidence": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            log_info_with_context(
                f"Successfully fetched {len(content)} characters from {url}",
                "Web Content"
            )
            return doc
            
        except Exception as e:
            log_error_with_traceback(e, f"Error fetching content from {url}")
            return None

    async def _load_chunks(self, source_path: str, source_type: str) -> List[str]:
        """Load and split source into chunks"""
        try:
            # Get shared progress instance
            progress = create_progress()
            
            # Create progress tasks
            load_task = progress.add_task("[cyan]Loading document...", total=1)
            chunk_task = progress.add_task("[cyan]Splitting into chunks...", total=None)
            
            # Process document
            processor = DocumentProcessor(self.llm)
            doc = await processor.process_document(source_path, source_type)
            
            if not doc.page_content:
                log_warning_with_context(f"No content extracted from {source_path}", "Document Loading")
                return []
                
            progress.update(load_task, advance=1)
            
            # Split content
            chunks = self.text_splitter.split_text(doc.page_content)
            progress.update(chunk_task, total=len(chunks), completed=len(chunks))
            
            # Log processing stats
            stats_table = Table(title="Document Processing")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            stats_table.add_row("Source", source_path)
            stats_table.add_row("Type", source_type)
            stats_table.add_row("Content Length", str(len(doc.page_content)))
            stats_table.add_row("Chunks", str(len(chunks)))
            
            if "analysis" in doc.metadata:
                analysis = doc.metadata["analysis"]
                if isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if isinstance(value, (str, int, float)):
                            stats_table.add_row(key, str(value))
                            
            console.print(stats_table)
            
            return chunks
            
        except Exception as e:
            log_error_with_traceback(e, f"Error loading chunks from {source_path}")
            return []

    async def _ensure_initialized(self) -> None:
        """Ensure all components are initialized"""
        if not self.llm or not self.embeddings or not self.vector_store or not self.graph:
            log_info_with_context("Initializing missing components", "Initialization")
            await self.initialize()

    async def _generate_embeddings(self, content: str) -> Optional[List[float]]:
        """Generate embeddings with proper error handling"""
        await self._ensure_initialized()
        if not self.embeddings:
            log_warning_with_context("Embeddings not initialized", "Embeddings")
            return None
        
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
            # Create parser with proper format instructions
            parser = PydanticOutputParser(pydantic_object=EntityResponse)
            format_instructions = parser.get_format_instructions()
            
            entity_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Extract entities from the given text. Return in JSON format.
Your response must be a valid JSON object that EXACTLY matches this schema:
{
    "entities": ["entity1", "entity2", ...]
}

IMPORTANT:
1. The entities field is required and must be an array of strings
2. Each entity should be a meaningful term or concept
3. Do not include any text before or after the JSON
4. Use proper JSON formatting with double quotes"""),
                HumanMessage(content=f"""
                    Text: {content}
                    {format_instructions}
                """)
            ])
            
            # Create chain with parser
            chain = entity_prompt | self.llm | parser
            
            # Execute chain
            result = await chain.ainvoke({})
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
            # Create parser with proper format instructions
            parser = PydanticOutputParser(pydantic_object=RelationshipResponse)
            format_instructions = parser.get_format_instructions()
            
            relationship_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Extract relationships between entities. Return in JSON format.
Your response must be a valid JSON object that EXACTLY matches this schema:
{
    "relationships": [
        {
            "source": "entity1",
            "relation": "is_a",
            "target": "entity2",
            "domain": "knowledge"
        }
    ]
}

IMPORTANT:
1. The relationships field is required and must be an array
2. Each relationship must have source, relation, target fields
3. relation must be EXACTLY one of: is_a, has_part, related_to
4. Do not include any text before or after the JSON
5. Use proper JSON formatting with double quotes"""),
                HumanMessage(content=f"""
                    Text: {content}
                    {format_instructions}
                """)
            ])
            
            # Create chain with parser
            chain = relationship_prompt | self.llm | parser
            
            # Execute chain
            result = await chain.ainvoke({})
            return result.relationships
            
        except Exception as e:
            log_error_with_traceback(e, "Error extracting relationships")
            return []

    async def _generate_metadata(self, content: str) -> SourceMetadata:
        """Generate metadata for content"""
        await self._ensure_initialized()
        if not self.llm:
            return self._create_default_metadata()
        
        try:
            # Create parser with proper format instructions
            parser = PydanticOutputParser(pydantic_object=MetadataResponse)
            format_instructions = parser.get_format_instructions()
            
            metadata_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Generate metadata for the content. Return in JSON format.
Your response must be a valid JSON object that EXACTLY matches this schema:
{
    "metadata": {
        "source_type": "text",
        "confidence_score": 0.8,
        "domain_relevance": 0.7,
        "timestamp": "2024-02-07T12:00:00Z",
        "validation_status": "pending",
        "domain": "knowledge"
    }
}

IMPORTANT:
1. The metadata field is required
2. source_type must be one of: text, pdf, web
3. confidence_score and domain_relevance must be between 0.0 and 1.0
4. timestamp must be in ISO format with timezone
5. validation_status must be one of: pending, processed, failed
6. Do not include any text before or after the JSON
7. Use proper JSON formatting with double quotes"""),
                HumanMessage(content=f"""
                    Text: {content}
                    {format_instructions}
                """)
            ])
            
            # Create chain with parser
            chain = metadata_prompt | self.llm | parser
            
            # Execute chain
            result = await chain.ainvoke({})
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

    async def generate_embeddings(self, state: ProcessingState) -> ProcessingState:
        """Generate embeddings in parallel"""
        if not self.embeddings:
            await self.initialize()

        tasks = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for task in state.tasks:
                if task.knowledge and not task.embeddings and not task.error:
                    if self.embeddings:
                        try:
                            futures.append(
                                executor.submit(
                                    lambda x: self.embeddings.embed_documents(x) if self.embeddings else None,
                                    [task.knowledge.content]
                                )
                            )
                        except Exception as e:
                            task.error = str(e)
                            futures.append(None)
                    else:
                        futures.append(None)
                else:
                    futures.append(None)

            for task, future in zip(state.tasks, futures):
                if future:
                    try:
                        embeddings = future.result()
                        task.embeddings = embeddings[0] if isinstance(embeddings, list) else embeddings
                    except Exception as e:
                        task.error = str(e)
                tasks.append(task)

        state.tasks = tasks
        return state