from typing import (
    List,
    Dict,
    Optional,
    Any,
    TypedDict,
    Protocol,
    cast,
    Awaitable,
    Callable,
    Union,
    TypeVar,
    Coroutine,
    Sequence
)
from datetime import datetime
import json
from pydantic import BaseModel, Field, validator, field_validator, SecretStr
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaEmbeddings
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
    DomainConfig,
    ConfidenceEvaluation,
    ConfidenceFactors
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
from scripts.chat_langchain import ChatLangChain
import os
from loguru import logger
import asyncio
from prompts.knowledge_acquisition import (
    get_knowledge_extraction_prompt,
    get_confidence_evaluation_prompt,
    EntityResponse,
    get_entity_extraction_prompt,
    RelationshipResponse,
    get_relationship_extraction_prompt,
    MetadataResponse,
    get_metadata_generation_prompt,
    get_plan_generation_prompt,
    get_join_decision_prompt
)

# Type variables for async functions
T = TypeVar('T')
AsyncFunc = Callable[..., Coroutine[Any, Any, T]]

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

    @field_validator("content")
    @classmethod
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
        try:
            # Initialize LLM first
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable must be set")
                
            llm = ChatLangChain(
                api_key=SecretStr(api_key),
                model="gemini-2.0-flash",
                temperature=0.7,
                pydantic_schema=EntityResponse
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
            
            log_info_with_context("Knowledge acquisition system initialized", "KnowledgeAcquisition", include_locals=True)
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize knowledge acquisition system", include_locals=True)
            raise

    async def initialize(self) -> "KnowledgeAcquisitionSystem":
        """Initialize vector store and graph database"""
        try:
            log_info_with_context("Starting system initialization", "KnowledgeAcquisition")
            
            # Initialize embeddings
            try:
                self.embeddings = OllamaEmbeddings(
                    model='bge-m3',  # Using a more reliable model
                    base_url='http://localhost:11434',

                )
                log_info_with_context("Embeddings initialized", "KnowledgeAcquisition")
            except Exception as e:
                log_error_with_traceback(e, "Failed to initialize embeddings", include_locals=True)
                raise
            
            # Initialize vector store
            if not self.vector_store:
                try:
                    self.vector_store = Chroma(
                        collection_name=self.config.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.config.persist_directory
                    )
                    log_info_with_context("Vector store initialized", "KnowledgeAcquisition")
                except Exception as e:
                    log_error_with_traceback(e, "Failed to initialize vector store", include_locals=True)
                    raise
            
            # Initialize graph database
            if not self.graph:
                try:
                    self.graph = Neo4jGraph(
                        url=self.config.neo4j_uri,
                        username=self.config.neo4j_username,
                        password=self.config.neo4j_password
                    )
                    log_info_with_context("Graph database initialized", "KnowledgeAcquisition")
                except Exception as e:
                    log_error_with_traceback(e, "Failed to initialize graph database", include_locals=True)
                    raise
            
            # Create necessary directories
            os.makedirs(self.config.persist_directory, exist_ok=True)
            os.makedirs("web", exist_ok=True)
            os.makedirs("downloads", exist_ok=True)
            
            return self
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize system components", include_locals=True)
            raise

    async def add_source(self, source_path: str, source_type: str) -> List[Document]:
        """Add a source to the knowledge base"""
        try:
            log_info_with_context(f"Processing source: {source_path}", "KnowledgeAcquisition")
            
            # Ensure system is initialized
            if not self.vector_store or not self.graph:
                await self.initialize()
            
            # Load and split content
            chunks = await self._load_chunks(source_path, source_type)
            if not chunks:
                log_warning_with_context(f"No content extracted from {source_path}", "KnowledgeAcquisition", include_locals=True)
                return []
            
            # Create progress display
            progress = create_progress()
            console.print(f"\n[cyan]Processing {len(chunks)} chunks from {source_path}[/cyan]")
            chunk_task = progress.add_task("[cyan]Processing chunks...", total=len(chunks))
            
            # Process chunks in parallel
            async def process_chunk(chunk: str) -> Optional[Document]:
                try:
                    # Process chunk
                    knowledge = await self.process_source(chunk)
                    if not knowledge:
                        return None
                        
                    # Generate embeddings (will be batched later)
                    embeddings = await self._generate_embeddings(chunk)
                    if not embeddings:
                        return None
                        
                    # Create document
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": source_path,
                            "type": source_type,
                            "knowledge": knowledge.model_dump() if hasattr(knowledge, "model_dump") else knowledge,
                            "embeddings": embeddings
                        }
                    )
                    progress.update(chunk_task, advance=1)
                    return doc
                    
                except Exception as e:
                    log_error_with_traceback(e, f"Failed to process chunk", include_locals=True)
                    return None
                    
            # Process chunks in parallel with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
            async def bounded_process(chunk: str) -> Optional[Document]:
                async with semaphore:
                    return await process_chunk(chunk)
                    
            # Process all chunks
            documents = []
            chunk_tasks = [bounded_process(chunk) for chunk in chunks]
            results = await asyncio.gather(*chunk_tasks)
            documents = [doc for doc in results if doc is not None]
            
            # Batch add to vector store
            if documents and self.vector_store is not None:
                try:
                    texts = [doc.page_content for doc in documents]
                    embeddings = [doc.metadata["embeddings"] for doc in documents]
                    metadatas = [{"source": doc.metadata["source"], "type": doc.metadata["type"]} for doc in documents]
                    
                    def batch_add_to_vector_store():
                        if self.vector_store is not None:
                            self.vector_store.add_texts(
                                texts=texts,
                                embeddings=embeddings,
                                metadatas=metadatas
                            )
                            
                    await asyncio.to_thread(batch_add_to_vector_store)
                    
                except Exception as e:
                    log_error_with_traceback(e, "Failed to batch add to vector store", include_locals=True)
                    
            # Batch add to graph
            if documents:
                try:
                    await asyncio.gather(*[
                        self._add_to_graph(doc.metadata["knowledge"])
                        for doc in documents
                    ])
                except Exception as e:
                    log_error_with_traceback(e, "Failed to batch add to graph", include_locals=True)
                    
            log_info_with_context(
                f"Processed {len(documents)} documents from {source_path}",
                "KnowledgeAcquisition",
                include_locals=True
            )
            return documents
            
        except Exception as e:
            log_error_with_traceback(e, f"Failed to add source: {source_path}", include_locals=True)
            raise

    async def process_source(self, content: str) -> ExtractedKnowledge:
        """Process a source and extract knowledge"""
        try:
            log_info_with_context("Starting source processing", "KnowledgeAcquisition")
            
            # Extract entities
            entities = await self._extract_entities(content)
            if not entities:
                log_warning_with_context("No entities extracted", "KnowledgeAcquisition", include_locals=True)
                entities = []
            
            # Extract relationships
            relationships = await self._extract_relationships(content)
            if not relationships:
                log_warning_with_context("No relationships extracted", "KnowledgeAcquisition", include_locals=True)
                relationships = []
            
            # Generate metadata
            metadata = await self._generate_metadata(content)
            if not metadata:
                log_warning_with_context("Failed to generate metadata", "KnowledgeAcquisition", include_locals=True)
                metadata = self._create_default_metadata()
            
            # Evaluate confidence
            confidence_eval = await self._evaluate_confidence(content, entities, relationships)
            confidence_score = (
                confidence_eval.confidence 
                if hasattr(confidence_eval, 'confidence') 
                else confidence_eval.factors.overall
                if hasattr(confidence_eval, 'factors')
                else 0.5
            )
            
            # Create knowledge object
            knowledge = ExtractedKnowledge(
                content=content,
                entities=entities,
                relationships=relationships,
                metadata=metadata,
                confidence=confidence_score
            )
            
            log_extraction_results(knowledge)
            return knowledge
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to process source", include_locals=True)
            raise

    async def _add_to_graph(self, knowledge: ExtractedKnowledge) -> None:
        """Add extracted knowledge to graph database"""
        try:
            if not self.graph:
                raise ValueError("Graph database not initialized")
            
            # Add entities
            for entity in knowledge.entities:
                try:
                    # Define synchronous function for graph operation
                    def add_entity() -> None:
                        if self.graph is not None:
                            self.graph.query(
                                f"MERGE (e:Entity {{name: $name, confidence: $confidence}})",
                                {"name": entity, "confidence": float(knowledge.confidence)}
                            )
                    
                    # Run in thread pool
                    await asyncio.to_thread(cast(Callable[[], None], add_entity))
                except Exception as e:
                    log_error_with_traceback(e, f"Failed to add entity: {entity}", include_locals=True)
                    continue
            
            # Add relationships
            for rel in knowledge.relationships:
                try:
                    # Define synchronous function for graph operation
                    def add_relationship() -> None:
                        if self.graph is not None:
                            self.graph.query(
                                """
                                MATCH (s:Entity {name: $source})
                                MATCH (t:Entity {name: $target})
                                MERGE (s)-[r:RELATES {type: $relation, confidence: $confidence}]->(t)
                                """,
                                {
                                    "source": rel.source,
                                    "target": rel.target,
                                    "relation": rel.relation,
                                    "confidence": float(knowledge.confidence)
                                }
                            )
                    
                    # Run in thread pool
                    await asyncio.to_thread(cast(Callable[[], None], add_relationship))
                except Exception as e:
                    log_error_with_traceback(e, f"Failed to add relationship: {rel}", include_locals=True)
                    continue
                    
        except Exception as e:
            log_error_with_traceback(e, "Failed to add knowledge to graph", include_locals=True)
            raise

    async def _generate_embeddings(self, content: str) -> Optional[List[float]]:
        """Generate embeddings for content"""
        try:
            if not self.embeddings:
                raise ValueError("Embeddings model not initialized")
            
            # Generate embeddings using a synchronous function
            def generate_embeddings() -> Optional[List[float]]:
                if not self.embeddings:
                    return None
                result = self.embeddings.embed_documents([content])
                return result[0] if result else None
            
            # Run in thread pool
            embeddings = await asyncio.to_thread(generate_embeddings)
            return embeddings
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to generate embeddings", include_locals=True)
            return None

    async def _extract_entities(self, content: str) -> List[str]:
        """Extract entities from content"""
        try:
            # First try using the entity extraction prompt
            prompt = get_entity_extraction_prompt()
            chain = prompt | self.llm | self.entity_parser
            result = await chain.ainvoke({"content": content})  # Note: changed text to content to match prompt
            entities = result.entities if hasattr(result, 'entities') else []
            
            # If no entities found, try using the knowledge extraction prompt as backup
            if not entities:
                knowledge_prompt = get_knowledge_extraction_prompt()
                knowledge_chain = knowledge_prompt | self.llm
                knowledge_result = await knowledge_chain.ainvoke({"text": content})
                if isinstance(knowledge_result, dict) and "entities" in knowledge_result:
                    entities = knowledge_result["entities"]
            
            # If still no entities, extract some basic ones from the content
            if not entities:
                # Split content into words and take unique non-stop words as entities
                words = set(content.split())
                stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by"}
                entities = [word for word in words if word.lower() not in stop_words and len(word) > 2][:5]
            
            # Ensure we have at least one entity
            if not entities:
                entities = ["general_concept"]
                
            return entities
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to extract entities", include_locals=True)
            # Return a default entity rather than empty list
            return ["general_concept"]

    async def _extract_relationships(self, content: str) -> List[Relationship]:
        """Extract relationships from content"""
        try:
            prompt = get_relationship_extraction_prompt()
            chain = prompt | self.llm | self.relationship_parser
            result = await chain.ainvoke({"text": content})
            return result.relationships if hasattr(result, 'relationships') else []
        except Exception as e:
            log_error_with_traceback(e, "Failed to extract relationships", include_locals=True)
            return []

    async def _generate_metadata(self, content: str) -> Optional[SourceMetadata]:
        """Generate metadata for content"""
        try:
            prompt = get_metadata_generation_prompt()
            chain = prompt | self.llm | self.metadata_parser
            result = await chain.ainvoke({"text": content})
            return result.metadata if hasattr(result, 'metadata') else None
        except Exception as e:
            log_error_with_traceback(e, "Failed to generate metadata", include_locals=True)
            return None

    def _create_default_metadata(self) -> SourceMetadata:
        """Create default metadata"""
        return SourceMetadata(
            source_type="text",
            confidence_score=0.5,
            domain_relevance=0.5,
            timestamp=datetime.now().isoformat(),
            validation_status="pending",
            domain=self.config.default_domain
        )

    async def _ensure_initialized(self) -> None:
        """Ensure all components are initialized"""
        if not self.llm or not self.embeddings or not self.vector_store or not self.graph:
            log_info_with_context("Initializing missing components", "Initialization")
            await self.initialize()

    async def _load_chunks(self, source_path: str, source_type: str = "text") -> List[str]:
        """Load and split content into chunks"""
        try:
            # Define synchronous loading functions
            def load_pdf() -> List[Document]:
                loader = PyPDFLoader(source_path)
                return loader.load()
            
            def load_text() -> List[Document]:
                loader = TextLoader(source_path)
                return loader.load()
            
            def split_documents(docs: List[Document]) -> List[str]:
                chunks = self.text_splitter.split_documents(docs)
                return [chunk.page_content for chunk in chunks]
            
            # Load content based on source type
            docs = await asyncio.to_thread(
                cast(Callable[[], List[Document]], load_pdf if source_type == "pdf" else load_text)
            )
            
            # Split into chunks
            chunks = await asyncio.to_thread(split_documents, docs)
            return chunks
            
        except Exception as e:
            log_error_with_traceback(e, f"Failed to load chunks from {source_path}", include_locals=True)
            return []

    async def _evaluate_confidence(self, content: str, entities: List[str], relationships: Sequence[Union[Relationship, Dict[str, Any]]]) -> ConfidenceEvaluation:
        """Evaluate confidence in extracted knowledge"""
        try:
            prompt = get_confidence_evaluation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=ConfidenceEvaluation)
            result = await chain.ainvoke({
                "content": content,
                "entities": entities,
                "relationships": [
                    rel.model_dump() if isinstance(rel, Relationship) else rel 
                    for rel in relationships
                ],
                "source_type": "text"
            })
            return result
        except Exception as e:
            log_error_with_traceback(e, "Failed to evaluate confidence", include_locals=True)
            return ConfidenceEvaluation(
                confidence=0.5,
                factors=ConfidenceFactors(),
                reasoning="Failed to evaluate confidence"
            )