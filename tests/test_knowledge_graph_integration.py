import asyncio
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest
from neo4j import GraphDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings

from knowledge_graph_runner import (
    DomainConfig,
    KnowledgeGraphSystem,
    KnowledgeNode,
    KnowledgeRelation,
    GraphRAGProcessor
)

# Test configuration
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687")
TEST_NEO4J_USER = os.getenv("TEST_NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD", "password")

@pytest.fixture(scope="session")
def domain_config():
    return DomainConfig(
        name="test_medical",
        entity_types=["Disease", "Drug", "Symptom"],
        relation_types=["TREATS", "CAUSES"],
        validation_rules={
            "confidence_threshold": 0.7,
            "required_sources": 1,
            "max_relation_distance": 2,
            "temporal_validity": {
                "publication_max_age_years": 5,
                "clinical_trial_status": ["ACTIVE", "COMPLETED"]
            }
        },
        search_strategies={
            "primary_sources": ["pubmed"],
            "search_depth": 2,
            "max_results_per_query": 10,
            "prioritize_recent": True,
            "include_preprints": False,
            "citation_threshold": 3
        }
    )

@pytest.fixture(scope="session")
def neo4j_driver():
    driver = GraphDatabase.driver(
        TEST_NEO4J_URI,
        auth=(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD)
    )
    yield driver
    driver.close()

@pytest.fixture(autouse=True)
def clean_neo4j(neo4j_driver):
    # Clean up before and after each test
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("DROP INDEX node_embeddings IF EXISTS")
    yield
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("DROP INDEX node_embeddings IF EXISTS")

@pytest.fixture
def knowledge_graph_system(domain_config):
    system = KnowledgeGraphSystem(domain_config)
    return system

@pytest.fixture
def embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@pytest.mark.asyncio
async def test_knowledge_acquisition_and_graph_rag(knowledge_graph_system, neo4j_driver):
    # Test basic knowledge acquisition with GraphRAG
    queries = [
        "What are the primary treatments for Type 2 Diabetes?",
        "What are the main side effects of Metformin?"
    ]
    
    results = await knowledge_graph_system.run(queries)
    
    # Verify results structure
    assert isinstance(results, dict)
    assert results["processed_documents"] > 0
    assert results["extracted_nodes"] > 0
    assert results["extracted_relationships"] > 0
    
    # Verify Neo4j graph population with embeddings
    with neo4j_driver.session() as session:
        # Check nodes have embeddings
        result = session.run("""
            MATCH (n)
            WHERE exists(n.embedding)
            RETURN count(n) as count
        """)
        assert result.single()["count"] > 0
        
        # Check relationships have descriptions
        result = session.run("""
            MATCH ()-[r]->()
            WHERE exists(r.description)
            RETURN count(r) as count
        """)
        assert result.single()["count"] > 0

@pytest.mark.asyncio
async def test_vector_search(knowledge_graph_system, neo4j_driver, embeddings):
    # Add test nodes with embeddings
    test_text = "Metformin is used to treat type 2 diabetes"
    test_embedding = embeddings.embed_query(test_text)
    
    node = KnowledgeNode(
        id="test_node",
        type="Drug",
        properties={"name": "Metformin", "description": test_text},
        confidence=0.9,
        sources=["test"],
        timestamp=datetime.now(),
        embedding=test_embedding
    )
    
    knowledge_graph_system.graph_manager.create_node(node)
    knowledge_graph_system.graph_manager.setup_vector_index()
    
    # Test vector similarity search
    query_text = "diabetes treatment medication"
    query_embedding = embeddings.embed_query(query_text)
    
    similar_nodes = knowledge_graph_system.graph_manager.find_similar_entities(
        embedding=query_embedding,
        k=1,
        similarity_cutoff=0.7
    )
    
    assert len(similar_nodes) > 0

@pytest.mark.asyncio
async def test_graph_rag_processing(knowledge_graph_system):
    # Test GraphRAG document processing
    test_text = """
    Metformin is a medication used to treat type 2 diabetes. It works by reducing glucose production
    in the liver and improving insulin sensitivity. Common side effects include nausea and diarrhea.
    """
    
    processor = knowledge_graph_system.graph_rag
    graph_docs = processor.process_text(test_text)
    
    # Verify document structure
    assert isinstance(graph_docs, list)
    assert len(graph_docs) > 0
    
    doc = graph_docs[0]
    assert "nodes" in doc
    assert "relationships" in doc
    
    # Verify node structure
    for node in doc["nodes"]:
        assert "id" in node
        assert "description" in node
        assert "embedding" in node
        assert isinstance(node["embedding"], list)
        assert len(node["embedding"]) > 0
        
    # Verify relationship structure
    for rel in doc["relationships"]:
        assert "source" in rel
        assert "target" in rel
        assert "type" in rel
        assert "description" in rel

@pytest.mark.asyncio
async def test_entity_merging(knowledge_graph_system, neo4j_driver):
    # Add similar entities
    text1 = "Metformin treats diabetes by reducing blood sugar"
    text2 = "Metformin is a medication that helps control blood glucose levels"
    
    processor = knowledge_graph_system.graph_rag
    docs1 = processor.process_text(text1)
    docs2 = processor.process_text(text2)
    
    # Populate graph
    knowledge_graph_system._populate_graph(docs1)
    knowledge_graph_system._populate_graph(docs2)
    knowledge_graph_system.graph_manager.setup_vector_index()
    
    # Run entity merging
    knowledge_graph_system._merge_similar_entities(similarity_cutoff=0.8)
    
    # Verify merged entities
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE n.properties.description CONTAINS 'Metformin'
            RETURN count(n) as count
        """)
        # Should have merged similar Metformin nodes
        assert result.single()["count"] == 1

@pytest.mark.asyncio
async def test_temporal_knowledge_tracking(knowledge_graph_system, neo4j_driver):
    # Add knowledge from different time periods
    old_text = "Old study shows metformin reduces blood sugar"
    new_text = "Recent study confirms metformin's effectiveness in diabetes treatment"
    
    processor = knowledge_graph_system.graph_rag
    old_docs = processor.process_text(old_text)
    new_docs = processor.process_text(new_text)
    
    # Modify timestamps
    for doc in old_docs:
        for node in doc["nodes"]:
            node["timestamp"] = (datetime.now() - timedelta(days=365*6)).isoformat()
            
    for doc in new_docs:
        for node in doc["nodes"]:
            node["timestamp"] = datetime.now().isoformat()
    
    # Populate graph
    knowledge_graph_system._populate_graph(old_docs)
    knowledge_graph_system._populate_graph(new_docs)
    
    # Verify temporal filtering
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE datetime(n.timestamp) > datetime() - duration('P5Y')
            RETURN count(n) as count
        """)
        recent_count = result.single()["count"]
        assert recent_count == len(new_docs[0]["nodes"])  # Only new nodes should be counted

@pytest.mark.asyncio
async def test_cross_domain_inference(knowledge_graph_system, neo4j_driver):
    # Add knowledge about diseases and drugs
    queries = [
        "What are the mechanisms of action for insulin in treating diabetes?",
        "How does metformin affect glucose metabolism?"
    ]
    
    await knowledge_graph_system.run(queries)
    
    # Verify cross-domain relationships
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (d:Disease)-[r]-(m:Drug)
            WHERE d.properties.description CONTAINS 'diabetes'
            AND m.properties.description CONTAINS 'metformin'
            RETURN count(r) as count
        """)
        cross_domain_count = result.single()["count"]
        assert cross_domain_count > 0

@pytest.mark.asyncio
async def test_knowledge_validation(knowledge_graph_system, neo4j_driver):
    # Add potentially contradictory information
    queries = [
        "What is the recommended daily dose of metformin for type 2 diabetes?",
        "What are the contraindications for metformin use?"
    ]
    
    await knowledge_graph_system.run(queries)
    
    # Verify source attribution and confidence scores
    with neo4j_driver.session() as session:
        # Check source attribution
        result = session.run("""
            MATCH (n)
            WHERE size(n.sources) >= 1
            RETURN count(n) as count
        """)
        sourced_count = result.single()["count"]
        assert sourced_count > 0
        
        # Check confidence scores
        result = session.run("""
            MATCH (n)
            WHERE exists(n.confidence)
            RETURN count(n) as count
        """)
        scored_count = result.single()["count"]
        assert scored_count > 0
        
        # Verify high-confidence relationships
        result = session.run("""
            MATCH ()-[r]->()
            WHERE r.confidence >= 0.7
            RETURN count(r) as count
        """)
        high_confidence_count = result.single()["count"]
        assert high_confidence_count > 0 