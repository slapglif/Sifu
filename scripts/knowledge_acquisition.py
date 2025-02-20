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
from scripts.text_web_browser_fixed import SimpleTextBrowser, web_search
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
from langchain.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup, Comment
import hashlib
from tqdm.rich import tqdm
import re
import google.api_core.exceptions

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
                model="gemini-1.5-flash",
                temperature=0.7,
                format="json",
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
            
            # Initialize parsers with function calling
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
                    base_url='http://localhost:11434'
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
            progress = await create_progress()
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
                    
                    # Convert knowledge to serializable format
                    if isinstance(knowledge, dict):
                        knowledge_dict = knowledge
                    elif hasattr(knowledge, "model_dump"):
                        knowledge_dict = knowledge.model_dump()
                    elif hasattr(knowledge, "dict"):
                        knowledge_dict = knowledge.dict()
                    else:
                        knowledge_dict = dict(knowledge)
                    
                    # Create document
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": source_path,
                            "type": source_type,
                            "knowledge": knowledge_dict,
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

    async def process_source(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a source document to extract knowledge."""
        try:
            logger.info(f"Starting source processing with content length: {len(content)}")
            
            # Preview content
            if isinstance(content, dict):
                logger.info(f"Content preview: {str(content)[:500]}...")
            else:
                logger.info(f"Content preview: {str(content)[:500]}...")
                
            # Extract entities
            logger.info("Extracting entities...")
            entities = await self._extract_entities(content)
            if not entities:
                logger.warning("No entities extracted")
                return {"content": content, "entities": [], "relationships": [], "metadata": self._create_default_metadata(), "confidence": 0.0}

            logger.info(f"Extracted and validated {len(entities)} entities: {entities}")
                
            # Extract relationships
            logger.info("Extracting relationships...")
            relationships = await self._extract_relationships(content, entities)
                
            # Generate metadata
            logger.info("Generating metadata...")
            source_metadata = self._create_default_metadata()
            if metadata:
                if isinstance(metadata, dict):
                    # Create new metadata object with combined fields
                    source_metadata = SourceMetadata(
                        **{
                            **source_metadata.model_dump(),
                            **metadata
                        }
                    )
                elif isinstance(metadata, SourceMetadata):
                    # Update fields from other metadata object
                    source_metadata = SourceMetadata(
                        **{
                            **source_metadata.model_dump(),
                            **metadata.model_dump()
                        }
                    )
            logger.info(f"Generated metadata: {source_metadata}")

            # Evaluate confidence
            logger.info("Evaluating confidence...")
            confidence = await self._evaluate_confidence(entities, relationships)
            logger.info(f"Confidence score: {confidence}")

            # Return full knowledge object
            return {
                "content": content,
                "entities": entities,
                "relationships": relationships,
                "metadata": source_metadata,
                "confidence": confidence
            }
        except Exception as e:
            log_error_with_traceback(e, "Error processing source")
            return {
                "content": content,
                "entities": [],
                "relationships": [],
                "metadata": self._create_default_metadata(),
                "confidence": 0.0
            }

    async def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        try:
            # Remove HTML
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
                
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
                
            # Get main content
            main_content = ""
            main_tags = soup.find_all(["article", "main", "div[role='main']"])
            if main_tags:
                main_content = " ".join(tag.get_text(strip=True, separator=" ") for tag in main_tags)
            else:
                body = soup.find("body")
                if body:
                    main_content = body.get_text(strip=True, separator=" ")
                else:
                    main_content = soup.get_text(strip=True, separator=" ")
                    
            # Clean up text
            main_content = re.sub(r'\s+', ' ', main_content)  # Normalize whitespace
            main_content = re.sub(r'\[.*?\]', '', main_content)  # Remove square bracket content
            main_content = re.sub(r'[^\x00-\x7F]+', '', main_content)  # Remove non-ASCII
            main_content = re.sub(r'(\w)\1{3,}', r'\1\1', main_content)  # Normalize repeated chars
            
            return main_content.strip()
            
        except Exception as e:
            log_error_with_traceback(e, "Error cleaning content")
            return content
            
    async def _add_to_graph(self, knowledge: ExtractedKnowledge) -> None:
        """Add extracted knowledge to graph database"""
        try:
            # Track graph operations
            graph_metrics = {
                "entities_added": 0,
                "relationships_added": 0,
                "failed_operations": 0
            }
            
            # Add entities first
            for entity in knowledge.entities:
                try:
                    # Define synchronous function for graph operation
                    def add_entity() -> None:
                        if self.graph is not None:
                            # Create entity with metadata
                            self.graph.query(
                                """
                                MERGE (e:Entity {name: $name})
                                SET e.domain = $domain,
                                    e.confidence = $confidence,
                                    e.timestamp = $timestamp,
                                    e.source = $source
                                """,
                                {
                                    "name": entity,
                                    "domain": knowledge.domain,
                                    "confidence": float(knowledge.confidence),
                                    "timestamp": datetime.now().isoformat(),
                                    "source": knowledge.metadata.source_type if hasattr(knowledge.metadata, "source_type") else "unknown"
                                }
                            )
                    
                    # Run in thread pool
                    await asyncio.to_thread(cast(Callable[[], None], add_entity))
                    graph_metrics["entities_added"] += 1
                    log_info_with_context(f"Added entity to graph: {entity}", "Graph")
                    
                except Exception as e:
                    log_error_with_traceback(e, f"Failed to add entity: {entity}", include_locals=True)
                    graph_metrics["failed_operations"] += 1
                    continue
            
            # Add relationships with retries
            max_retries = 3
            for rel in knowledge.relationships:
                success = False
                for attempt in range(max_retries):
                    try:
                        # Convert dict to Relationship if needed
                        if isinstance(rel, dict):
                            rel = Relationship(**rel)
                        
                        # Define synchronous function for graph operation
                        def add_relationship() -> None:
                            if self.graph is not None:
                                # First ensure entities exist
                                self.graph.query(
                                    """
                                    MERGE (s:Entity {name: $source})
                                    ON CREATE SET s.domain = $domain,
                                                s.confidence = $confidence,
                                                s.timestamp = $timestamp
                                    MERGE (t:Entity {name: $target})
                                    ON CREATE SET t.domain = $domain,
                                                t.confidence = $confidence,
                                                t.timestamp = $timestamp
                                    """,
                                    {
                                        "source": rel.source,
                                        "target": rel.target,
                                        "domain": knowledge.domain,
                                        "confidence": float(knowledge.confidence),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                )
                                
                                # Then create relationship with metadata
                                self.graph.query(
                                    """
                                    MATCH (s:Entity {name: $source})
                                    MATCH (t:Entity {name: $target})
                                    MERGE (s)-[r:RELATES {type: $relation}]->(t)
                                    SET r.confidence = $confidence,
                                        r.domain = $domain,
                                        r.timestamp = $timestamp,
                                        r.source = $source_type,
                                        r.bidirectional = $bidirectional
                                    """,
                                    {
                                        "source": rel.source,
                                        "target": rel.target,
                                        "relation": rel.relation,
                                        "confidence": float(knowledge.confidence),
                                        "domain": knowledge.domain,
                                        "timestamp": datetime.now().isoformat(),
                                        "source_type": knowledge.metadata.source_type if hasattr(knowledge.metadata, "source_type") else "unknown",
                                        "bidirectional": rel.relation in ["similar_to", "related_to"]
                                    }
                                )
                        
                        # Run in thread pool
                        await asyncio.to_thread(cast(Callable[[], None], add_relationship))
                        graph_metrics["relationships_added"] += 1
                        log_info_with_context(f"Added relationship to graph: {rel.source} -{rel.relation}-> {rel.target}", "Graph")
                        success = True
                        break
                        
                    except Exception as e:
                        if attempt == max_retries - 1:
                            log_error_with_traceback(e, f"Failed to add relationship after {max_retries} attempts: {rel}", include_locals=True)
                            graph_metrics["failed_operations"] += 1
                        await asyncio.sleep(1)  # Wait before retry
                        continue
                
                if not success:
                    continue
            
            # Log final metrics
            log_info_with_context(
                f"Graph operations completed - Added {graph_metrics['entities_added']} entities, "
                f"{graph_metrics['relationships_added']} relationships, "
                f"{graph_metrics['failed_operations']} operations failed",
                "Graph"
            )
            
            # Store metrics in knowledge metadata
            if hasattr(knowledge.metadata, "graph_metrics"):
                knowledge.metadata.graph_metrics = graph_metrics
                    
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
            # Clean HTML content first
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove navigation, headers, footers etc
            for element in soup(["nav", "header", "footer", "aside", "script", "style"]):
                element.decompose()
            
            # Get clean text
            clean_content = soup.get_text(separator=" ", strip=True)
            
            # First try using the entity extraction prompt
            prompt = get_entity_extraction_prompt()
            chain = prompt | self.llm | self.entity_parser
            result = await chain.ainvoke({"content": clean_content})
            entities = result.entities if hasattr(result, 'entities') else []
            
            # If no entities found, try using the knowledge extraction prompt as backup
            if not entities:
                knowledge_prompt = get_knowledge_extraction_prompt()
                knowledge_chain = knowledge_prompt | self.llm
                knowledge_result = await knowledge_chain.ainvoke({"text": clean_content})
                if isinstance(knowledge_result, dict) and "entities" in knowledge_result:
                    entities = knowledge_result["entities"]
            
            # If still no entities, extract some basic ones from the content
            if not entities:
                # Extract capitalized phrases and medical terms
                words = clean_content.split()
                phrases = []
                current_phrase = []
                
                medical_indicators = [
                    "disease", "syndrome", "disorder", "treatment", "therapy",
                    "medicine", "drug", "symptom", "patient", "clinical",
                    "medical", "health", "brain", "neural", "immune",
                    "inflammation", "gut", "vagus", "nerve", "axis"
                ]
                
                for word in words:
                    # Check if word starts with capital letter
                    if word[0].isupper() or any(ind in word.lower() for ind in medical_indicators):
                        current_phrase.append(word)
                    elif current_phrase:
                        if len(current_phrase) > 0:
                            phrases.append(" ".join(current_phrase))
                        current_phrase = []
                        
                # Add any remaining phrase
                if current_phrase:
                    phrases.append(" ".join(current_phrase))
                    
                # Use extracted phrases as entities
                entities = list(set(phrases))[:10]  # Limit to top 10 unique entities
                
            # Ensure we have at least one entity
            if not entities:
                entities = ["general_concept"]
                
            return entities
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to extract entities", include_locals=True)
            # Return a default entity rather than empty list
            return ["general_concept"]

    async def _validate_entities(self, entities: List[str], domain: str) -> List[str]:
        """Validate extracted entities against domain ontology."""
        try:
            # Get domain config
            domain_config = next(
                (d for d in self.config.domains if d.name == domain),
                next(iter(self.config.domains))  # Fallback to first domain
            )
            
            # Get valid entity classes from ontology
            valid_classes = {cls.name.lower(): cls for cls in domain_config.classes}
            
            # Track validated entities
            validated = []
            for entity in entities:
                # Clean entity text
                clean_entity = entity.strip()
                if not clean_entity:
                    continue
                    
                # Check if entity matches any ontology class
                entity_type = None
                for class_name, cls in valid_classes.items():
                    # Check direct match
                    if clean_entity.lower() == class_name:
                        entity_type = class_name
                        break
                        
                    # Check if class name appears in entity
                    if class_name in clean_entity.lower():
                        entity_type = class_name
                        break
                        
                    # Check properties
                    for prop in cls.properties:
                        if prop.lower() in clean_entity.lower():
                            entity_type = class_name
                            break
                            
                    if entity_type:
                        break
                        
                # Validate entity
                if entity_type:
                    # Entity matches ontology - keep as is
                    validated.append(clean_entity)
                else:
                    # Try to map to ontology
                    mapped_entity = await self._map_to_ontology(clean_entity, valid_classes)
                    if mapped_entity:
                        validated.append(mapped_entity)
                    else:
                        # Keep original entity if it seems medical/scientific
                        medical_indicators = [
                            "disease", "syndrome", "disorder", "treatment", "therapy",
                            "medicine", "drug", "symptom", "patient", "clinical",
                            "medical", "health", "brain", "neural", "immune",
                            "inflammation", "gut", "vagus", "nerve", "axis"
                        ]
                        if any(ind in clean_entity.lower() for ind in medical_indicators):
                            validated.append(clean_entity)
                    
            # Ensure we have at least one entity
            if not validated:
                validated = ["general_medical_concept"]
                
            return validated
            
        except Exception as e:
            log_error_with_traceback(e, "Error validating entities")
            return entities  # Return original list on error

    async def _map_to_ontology(self, entity: str, valid_classes: Dict[str, Any]) -> Optional[str]:
        """Map an entity to the closest matching ontology class."""
        try:
            # Create mapping prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Map the given entity to the most appropriate ontology class.
The entity should be mapped to one of the following classes:
{classes}

Rules:
1. Only map if there is a clear semantic match
2. Consider synonyms and related terms
3. Return null if no good match exists
4. Maintain proper medical terminology
5. Preserve specificity where possible"""),
                ("human", "Entity to map: {entity}")
            ])
            
            # Get mapping
            result = await self.llm.ainvoke(
                prompt.format_messages(
                    classes="\n".join(f"- {name}: {cls.description}" for name, cls in valid_classes.items()),
                    entity=entity
                )
            )
            
            # Parse result
            if result and isinstance(result, str):
                mapped = result.strip()
                if mapped.lower() in valid_classes:
                    return mapped
                    
            return None
            
        except Exception as e:
            log_error_with_traceback(e, "Error mapping to ontology")
            return None

    async def _extract_relationships(self, content: str, entities: List[str]) -> List[Relationship]:
        """Extract relationships from content"""
        try:
            prompt = get_relationship_extraction_prompt()
            chain = prompt | self.llm | self.relationship_parser
            result = await chain.ainvoke({"content": content, "entities": entities})
            relationships = result.relationships if hasattr(result, 'relationships') else []
            
            # Convert any dict relationships to Relationship objects
            converted_relationships = []
            for rel in relationships:
                try:
                    if isinstance(rel, dict):
                        # Map non-standard relation types to standard ones
                        if 'relation' in rel:
                            relation = rel['relation'].lower()
                            # Map common variations
                            relation_mapping = {
                                # Methodology relationships
                                'used_for': 'uses',
                                'used_in': 'uses',
                                'utilizes': 'uses',
                                'requires': 'uses',
                                'needs': 'uses',
                                'depends_on': 'uses',
                                'applied_to': 'applies',
                                'applied_in': 'applies',
                                'applied_for': 'applies',
                                'implemented_by': 'implements',
                                'implemented_in': 'implements',
                                'implemented_with': 'implements',
                                
                                # Performance relationships
                                'enhances': 'improves',
                                'boosts': 'improves',
                                'increases': 'improves',
                                'optimizes': 'improves',
                                'performs_better': 'outperforms',
                                'better_performance': 'outperforms',
                                'superior_to': 'outperforms',
                                'reaches': 'achieves',
                                'attains': 'achieves',
                                'accomplishes': 'achieves',
                                
                                # Component relationships
                                'includes': 'contains',
                                'incorporates': 'contains',
                                'encompasses': 'contains',
                                'made_of': 'consists_of',
                                'composed_of': 'consists_of',
                                'comprised_of': 'consists_of',
                                'belongs_to': 'part_of',
                                'member_of': 'part_of',
                                'element_of': 'part_of',
                                
                                # Comparison relationships
                                'superior_to': 'better_than',
                                'exceeds': 'better_than',
                                'outranks': 'better_than',
                                'resembles': 'similar_to',
                                'like': 'similar_to',
                                'analogous_to': 'similar_to',
                                'differs_from': 'different_from',
                                'unlike': 'different_from',
                                'distinct_from': 'different_from',
                                
                                # Causal relationships
                                'results_in': 'leads_to',
                                'produces': 'causes',
                                'creates': 'causes',
                                'generates': 'causes',
                                'impacts': 'affects',
                                'influences': 'affects',
                                'modifies': 'affects',
                                
                                # Temporal relationships
                                'comes_before': 'precedes',
                                'before': 'precedes',
                                'prior_to': 'precedes',
                                'comes_after': 'follows',
                                'after': 'follows',
                                'subsequent_to': 'follows',
                                'happens_with': 'concurrent_with',
                                'simultaneous_with': 'concurrent_with',
                                'parallel_to': 'concurrent_with',
                                
                                # Legacy relationships
                                'type_of': 'is_a',
                                'kind_of': 'is_a',
                                'instance_of': 'is_a',
                                'contains_part': 'has_part',
                                'includes_part': 'has_part',
                                'possesses': 'has_part',
                                'connected_to': 'related_to',
                                'associated_with': 'related_to',
                                'linked_to': 'related_to'
                            }
                            
                            # Try to map the relation, fallback to related_to if no mapping found
                            rel['relation'] = relation_mapping.get(relation, 'related_to')
                            
                            # If relation is still not valid, use related_to as fallback
                            valid_relations = [
                                'uses', 'applies', 'implements',
                                'improves', 'outperforms', 'achieves',
                                'contains', 'consists_of', 'part_of',
                                'better_than', 'similar_to', 'different_from',
                                'leads_to', 'causes', 'affects',
                                'precedes', 'follows', 'concurrent_with',
                                'is_a', 'has_part', 'related_to'
                            ]
                            if rel['relation'] not in valid_relations:
                                log_warning_with_context(
                                    f"Invalid relation type '{rel['relation']}' mapped to 'related_to'",
                                    "Relationship Extraction"
                                )
                                rel['relation'] = 'related_to'
                            
                        # Validate source and target
                        if not rel.get('source') or not rel.get('target'):
                            log_warning_with_context(
                                f"Missing source or target in relationship: {rel}",
                                "Relationship Extraction"
                            )
                            continue
                            
                        # Validate relationship makes sense
                        if rel['source'] == rel['target']:
                            log_warning_with_context(
                                f"Self-referential relationship detected: {rel}",
                                "Relationship Extraction"
                            )
                            continue
                            
                        # Create Relationship object with mapped relation
                        converted_rel = Relationship(**rel)
                        
                        # Evaluate relationship confidence
                        confidence = await self._evaluate_relationship_confidence(
                            converted_rel,
                            content
                        )
                        converted_rel.confidence = confidence
                        
                        # Only add if confidence meets threshold
                        if confidence >= self.config.confidence_thresholds.get("relationship", 0.3):
                            converted_relationships.append(converted_rel)
                        else:
                            log_warning_with_context(
                                f"Relationship confidence too low: {confidence}",
                                "Relationship Extraction"
                            )
                    else:
                        converted_relationships.append(rel)
                except Exception as e:
                    log_error_with_traceback(e, f"Failed to convert relationship: {rel}", include_locals=True)
                    continue
            
            return converted_relationships
        except Exception as e:
            log_error_with_traceback(e, "Failed to extract relationships", include_locals=True)
            return []

    async def _evaluate_relationship_confidence(self, relationship: Relationship, context: str) -> float:
        """Evaluate confidence in a relationship."""
        try:
            # Check if relationship type is appropriate for entities
            type_confidence = 0.8  # Base confidence for valid relationship type
            
            # Check if entities are mentioned close to each other in context
            proximity_confidence = 0.0
            source_pos = context.lower().find(relationship.source.lower())
            target_pos = context.lower().find(relationship.target.lower())
            if source_pos >= 0 and target_pos >= 0:
                distance = abs(source_pos - target_pos)
                # Higher confidence for closer entities (more lenient distance scaling)
                proximity_confidence = max(0.0, 1.0 - (distance / 2000))  # Increased from 1000
            
            # Check if relationship is explicitly stated
            explicit_confidence = 0.0
            relation_terms = {
                'uses': ['uses', 'utilizing', 'employs', 'requires'],
                'applies': ['applies', 'applying', 'application'],
                'implements': ['implements', 'implementation'],
                'improves': ['improves', 'enhances', 'boosts'],
                'outperforms': ['outperforms', 'better than', 'superior to'],
                'achieves': ['achieves', 'attains', 'reaches'],
                'contains': ['contains', 'includes', 'incorporates'],
                'consists_of': ['consists of', 'made of', 'composed of'],
                'part_of': ['part of', 'belongs to', 'member of'],
                'better_than': ['better than', 'superior to', 'exceeds'],
                'similar_to': ['similar to', 'like', 'resembles'],
                'different_from': ['different from', 'unlike', 'distinct from'],
                'leads_to': ['leads to', 'results in', 'causes'],
                'causes': ['causes', 'produces', 'creates'],
                'affects': ['affects', 'impacts', 'influences'],
                'precedes': ['precedes', 'before', 'prior to'],
                'follows': ['follows', 'after', 'subsequent to'],
                'concurrent_with': ['concurrent with', 'simultaneous', 'parallel'],
                'is_a': ['is a', 'type of', 'kind of'],
                'has_part': ['has part', 'contains part', 'includes part'],
                'related_to': ['related to', 'connected to', 'associated with']
            }
            
            # Look for explicit relationship terms
            terms = relation_terms.get(relationship.relation, [])
            for term in terms:
                if term.lower() in context.lower():
                    explicit_confidence = 0.9
                    break
            
            # If no explicit terms found but entities are close, give some confidence
            if explicit_confidence == 0.0 and proximity_confidence > 0.5:
                explicit_confidence = 0.4  # Added base confidence for proximity
                
            # Calculate overall confidence with adjusted weights
            confidence = (
                type_confidence * 0.4 +  # Increased from 0.33
                proximity_confidence * 0.3 +  # Same weight
                explicit_confidence * 0.3  # Decreased from 0.33
            )
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            log_error_with_traceback(e, "Error evaluating relationship confidence")
            return 0.5  # Default confidence on error

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
        """Create default metadata object."""
        return SourceMetadata(
            source_type="text",
            confidence_score=0.8,  # Increased from 0.5
            domain_relevance=0.8,  # Increased from 0.5
            timestamp=datetime.now().isoformat(),
            validation_status="pending",
            domain=self.config.domains[0].name if self.config.domains else "medical"  # Use first configured domain or fallback to medical
        )

    async def _ensure_initialized(self) -> None:
        """Ensure all components are initialized"""
        if not self.llm or not self.embeddings or not self.vector_store or not self.graph:
            log_info_with_context("Initializing missing components", "Initialization")
            await self.initialize()

    async def _load_chunks(self, source_path: str, source_type: str = "text") -> List[str]:
        """Load and split content into chunks"""
        try:
            # Load content
            loader = TextLoader(source_path)
            docs = await asyncio.to_thread(loader.load)
            
            # Split into chunks with metadata
            chunks = []
            for doc in tqdm(docs, desc="Processing documents"):
                # Get metadata
                metadata = doc.metadata.copy()
                
                # Split content
                text_chunks = self.text_splitter.split_text(doc.page_content)
                
                # Create chunks with metadata
                for i, chunk in enumerate(text_chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": f"{metadata.get('source_id', 'unknown')}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    })
                    chunks.append(Document(page_content=chunk, metadata=chunk_metadata))
            
            return [chunk.page_content for chunk in chunks]
            
        except Exception as e:
            log_error_with_traceback(e, f"Failed to load chunks from {source_path}", include_locals=True)
            return []

    async def _evaluate_confidence(self, entities: List[str], relationships: List[Relationship]) -> float:
        """Evaluate confidence in extracted knowledge"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                prompt = get_confidence_evaluation_prompt()
                chain = prompt | self.llm | PydanticOutputParser(pydantic_object=ConfidenceEvaluation)
                
                # Convert relationships to dicts
                rel_dicts = []
                for rel in relationships:
                    if isinstance(rel, dict):
                        rel_dicts.append(rel)
                    else:
                        # Assuming rel is a Pydantic model
                        rel_dicts.append({
                            'source': rel.source,
                            'relation': rel.relation,
                            'target': rel.target,
                            'confidence': getattr(rel, 'confidence', 1.0)
                        })
                
                result = await chain.ainvoke({
                    "entities": entities,
                    "relationships": rel_dicts,
                    "source_type": "text"
                })
                return result.confidence
            except Exception as e:
                if isinstance(e, google.api_core.exceptions.ResourceExhausted):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        log_warning_with_context(
                            f"Rate limit hit, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})",
                            "Confidence Evaluation"
                        )
                        await asyncio.sleep(delay)
                        continue
                
                # If we've exhausted retries or hit a different error, fall back to calculated confidence
                log_error_with_traceback(e, "Failed to evaluate confidence", include_locals=True)
                
                # Calculate fallback confidence based on available data
                entity_confidence = min(1.0, len(entities) * 0.2) if entities else 0.3
                relationship_confidence = min(1.0, len(relationships) * 0.2) if relationships else 0.3
                
                # Weight the components
                confidence = (entity_confidence * 0.6) + (relationship_confidence * 0.4)
                
                log_warning_with_context(
                    f"Using fallback confidence calculation: {confidence:.2f}",
                    "Confidence Evaluation"
                )
                return confidence
        
        # If we somehow exit the loop without returning, use a safe default
        return 0.5