# Knowledge Graph Generation System

A flexible, domain-agnostic knowledge graph generation system that combines RAG, synthetic knowledge generation, and fine-tuning capabilities.

## Features

1. **Multi-Modal Knowledge Acquisition**
   - Configurable search across multiple sources
   - Domain-adaptive search strategies
   - Multi-format document processing
   - Pluggable source interfaces

2. **Question-Answer System**
   - Dynamic question generation
   - Self-questioning for validation
   - Answer synthesis with source validation
   - Iterative refinement loops

3. **Synthetic Knowledge Generation**
   - Novel insight generation
   - Cross-reference validation
   - Hypothesis generation
   - Pattern recognition
   - Confidence scoring

4. **LoRA Training Integration**
   - Domain-specific adaptation
   - Continuous learning
   - Parameter-efficient fine-tuning
   - Performance monitoring

5. **Knowledge Graph Evolution**
   - Dynamic schema adaptation
   - Relationship strength weighting
   - Temporal knowledge tracking
   - Source attribution
   - Contradiction resolution

## Prerequisites

1. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Neo4j Database**
   - Install Neo4j Desktop or use Docker:
     ```bash
     docker run \
       --name neo4j \
       -p 7474:7474 -p 7687:7687 \
       -e NEO4J_AUTH=neo4j/password \
       -e NEO4J_PLUGINS='["apoc"]' \
       neo4j:latest
     ```

3. **Ollama Setup**
   - Install Ollama: https://ollama.ai/
   - Pull the Falcon model:
     ```bash
     ollama pull falcon3:3b
     ```

## Configuration

1. **Environment Variables**
   Create a `.env` file:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   OLLAMA_BASE_URL=http://localhost:11434
   ```

2. **Domain Configuration**
   Create a domain config JSON file (see `sample_domain_config.json`):
   ```json
   {
     "name": "your_domain",
     "entity_types": [...],
     "relation_types": [...],
     "validation_rules": {...},
     "search_strategies": {...}
   }
   ```

3. **Initial Queries**
   Create a queries JSON file (see `sample_queries.json`):
   ```json
   [
     "What are the latest treatments for...",
     "Identify genetic pathways involved in...",
     ...
   ]
   ```

## Usage

1. **Run the System**
   ```bash
   python knowledge_graph_runner.py \
     --domain-config path/to/domain_config.json \
     --initial-queries path/to/queries.json
   ```

2. **Run Integration Tests**
   ```bash
   # Start Neo4j test instance
   docker run \
     --name neo4j-test \
     -p 7475:7474 -p 7688:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:latest

   # Run tests
   pytest tests/test_knowledge_graph_integration.py -v
   ```

## Development

1. **Adding New Knowledge Sources**
   - Implement new tools in `MultiModalKnowledgeAcquisition`
   - Add source configuration to domain config

2. **Custom Model Adaptation**
   - Modify LoRA configuration in `LoRATrainer`
   - Adjust training parameters

3. **Schema Evolution**
   - Update domain config
   - System will automatically adapt

## Integration Testing

The system includes comprehensive integration tests that verify:

1. Knowledge Acquisition
   - Source retrieval
   - Content extraction
   - Knowledge structuring

2. Synthetic Knowledge
   - Pattern recognition
   - Hypothesis generation
   - Confidence scoring

3. Temporal Tracking
   - Time-based filtering
   - Source freshness
   - Historical tracking

4. Model Adaptation
   - Training effectiveness
   - Domain specialization
   - Performance metrics

## Troubleshooting

1. **Neo4j Connection Issues**
   ```bash
   # Check Neo4j status
   docker ps
   docker logs neo4j
   ```

2. **Ollama Issues**
   ```bash
   # Check Ollama service
   ollama ps
   ollama logs
   ```

3. **Model Adaptation**
   - Check CUDA availability
   - Verify VRAM requirements
   - Monitor training logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - See LICENSE file for details 