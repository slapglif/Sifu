from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

class OntologyRelation(BaseModel):
    """Model for ontology relationships"""
    name: str = Field(description="Name of the relationship type")
    description: str = Field(description="Description of what this relationship means")
    symmetric: bool = Field(default=False, description="Whether the relationship is symmetric (a->b means b->a)")
    transitive: bool = Field(default=False, description="Whether the relationship is transitive (a->b->c means a->c)")
    inverse_of: Optional[str] = Field(default=None, description="Name of the inverse relationship if any")

class OntologyClass(BaseModel):
    """Model for ontology classes"""
    name: str = Field(description="Name of the class")
    description: str = Field(description="Description of what this class represents")
    parent_classes: List[str] = Field(default_factory=list, description="Parent classes in the hierarchy")
    properties: Dict[str, str] = Field(default_factory=dict, description="Properties and their types")
    allowed_relations: List[str] = Field(default_factory=list, description="Allowed relationship types for this class")

class DomainConfig(BaseModel):
    """Configuration for a specific knowledge domain"""
    name: str = Field(description="Name of the domain")
    description: str = Field(description="Description of the domain")
    classes: List[OntologyClass] = Field(description="Classes in this domain's ontology")
    relations: List[OntologyRelation] = Field(description="Relationship types in this domain")
    validation_rules: Dict[str, Dict] = Field(description="Validation rules for the domain")
    confidence_thresholds: Dict[str, float] = Field(description="Confidence thresholds for different operations")
    search_strategies: List[str] = Field(description="Search strategies for this domain")

class KnowledgeAcquisitionConfig(BaseModel):
    """Configuration for knowledge acquisition"""
    domains: List[DomainConfig] = Field(description="Configured knowledge domains")
    default_domain: str = Field(description="Default domain to use")
    source_types: List[str] = Field(description="Allowed source types")
    validation_rules: Dict = Field(description="Global validation rules")
    confidence_thresholds: Dict[str, float] = Field(description="Global confidence thresholds")
    chunk_size: int = Field(default=2000, description="Size of text chunks for processing")
    chunk_overlap: int = Field(default=400, description="Overlap between text chunks")
    max_tokens: int = Field(default=32000, description="Maximum tokens for model context")
    enable_web_search: bool = Field(default=False, description="Whether to enable web search enrichment")
    collection_name: str = Field(default="knowledge_base", description="Name of the vector store collection")
    persist_directory: str = Field(default="./data/vector_store", description="Directory to persist vector store")
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")

    @validator("domains")
    def validate_domains(cls, v):
        domain_names = [d.name for d in v]
        if len(domain_names) != len(set(domain_names)):
            raise ValueError("Domain names must be unique")
        return v

    @validator("default_domain")
    def validate_default_domain(cls, v, values):
        if "domains" in values and v not in [d.name for d in values["domains"]]:
            raise ValueError("Default domain must be one of the configured domains")
        return v

class Relationship(BaseModel):
    """Schema for knowledge relationships"""
    source: str = Field(description="Source entity")
    relation: Literal["is_a", "has_part", "related_to"] = Field(description="Type of relationship")
    target: str = Field(description="Target entity")
    domain: str = Field(default="knowledge", description="Domain this relationship belongs to")
    confidence: float = Field(default=1.0, description="Confidence in this relationship", ge=0.0, le=1.0)

    @validator("relation")
    def validate_relation(cls, v):
        valid_relations = ["is_a", "has_part", "related_to"]
        if v not in valid_relations:
            raise ValueError(f"relation must be one of: {valid_relations}")
        return v

class SourceMetadata(BaseModel):
    """Model for source metadata"""
    source_type: str = Field(description="Type of source (academic, web, etc)")
    confidence_score: float = Field(description="Confidence in source reliability")
    domain_relevance: float = Field(description="Relevance to current domain")
    timestamp: str = Field(description="When the source was processed")
    validation_status: str = Field(description="Validation status of the source")
    domain: str = Field(default="knowledge", description="Domain this source belongs to")

    @validator("timestamp")
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("timestamp must be in ISO format")

    @validator("validation_status")
    def validate_status(cls, v):
        valid_statuses = ["pending", "processed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"validation_status must be one of: {valid_statuses}")
        return v

    @validator("source_type")
    def validate_source_type(cls, v):
        valid_types = ["text", "pdf", "web"]
        if v not in valid_types:
            raise ValueError(f"source_type must be one of: {valid_types}")
        return v

class ExtractedKnowledge(BaseModel):
    """Model for extracted knowledge"""
    content: str = Field(description="The content or summary of the source")
    entities: List[str] = Field(description="List of extracted entities")
    relationships: List[Relationship] = Field(description="List of relationships between entities")
    confidence: float = Field(default=1.0, description="Overall confidence score for the extraction", ge=0.0, le=1.0)
    metadata: Optional[SourceMetadata] = Field(None, description="Additional metadata about the extraction")
    domain: str = Field(default="knowledge", description="Domain this knowledge belongs to")

    @validator("content")
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()

    @validator("entities")
    def validate_entities(cls, v):
        if not v:
            raise ValueError("entities list cannot be empty")
        return [e.strip() for e in v if e.strip()]

    @validator("relationships")
    def validate_relationships(cls, v):
        if not v:
            raise ValueError("relationships list cannot be empty")
        return v

class LLMResponse(BaseModel):
    """Model for LLM responses"""
    content: str = Field(description="Response content")
    model: Optional[str] = Field(None, description="Model used")
    usage: Optional[Dict] = Field(None, description="Token usage info")

class EntityExtractionInput(BaseModel):
    """Input for entity extraction"""
    text: str = Field(description="Text to extract entities from")
    domain: str = Field(default="knowledge", description="Domain for extraction")

class RelationshipExtractionInput(BaseModel):
    """Input for relationship extraction"""
    text: str = Field(description="Text to extract relationships from")
    entities: List[str] = Field(description="Known entities")
    domain: str = Field(default="knowledge", description="Domain for extraction")

class MetadataGenerationInput(BaseModel):
    """Input for metadata generation"""
    text: str = Field(description="Text to generate metadata for")
    domain: str = Field(default="knowledge", description="Domain for metadata")

class DocumentProcessingInput(BaseModel):
    """Input for document processing"""
    source_path: str = Field(description="Path to source document")
    source_type: str = Field(description="Type of source")
    domain: str = Field(default="knowledge", description="Domain for processing")

class ConfidenceFactors(BaseModel):
    """Model for confidence evaluation factors"""
    content_quality: float = Field(default=0.5, description="Quality of the content", ge=0.0, le=1.0)
    entity_confidence: float = Field(default=0.5, description="Confidence in entity extraction", ge=0.0, le=1.0)
    relationship_validity: float = Field(default=0.5, description="Validity of relationships", ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.5, description="Reliability of the source", ge=0.0, le=1.0)
    context_relevance: float = Field(default=0.5, description="Relevance to context", ge=0.0, le=1.0)
    overall: float = Field(default=0.5, description="Overall confidence score", ge=0.0, le=1.0)

    def get(self, key: str, default: float = 0.5) -> float:
        """Get factor value with default"""
        return getattr(self, key, default)

class ConfidenceEvaluation(BaseModel):
    """Model for confidence evaluation"""
    confidence: float = Field(description="Overall confidence score", ge=0.0, le=1.0)
    factors: ConfidenceFactors = Field(description="Detailed confidence factors")
    reasoning: str = Field(description="Reasoning behind confidence evaluation") 