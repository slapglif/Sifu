"""Confidence evaluation prompts."""

CONFIDENCE_EVALUATION_SYSTEM = """You are a confidence evaluation expert. Analyze the extracted knowledge and provide a confidence score.
Your response must be a valid JSON object with these fields:
{{{{
    "confidence": 0.85,  # Between 0.0 and 1.0
    "reasoning": "Detailed explanation of the confidence score",
    "factors": {{{{
        "content_quality": 0.85,  # Quality and coherence of content
        "entity_confidence": 0.85,  # Confidence in entity extraction
        "relationship_validity": 0.85,  # Validity of relationships
        "source_reliability": 0.85,  # Reliability of the source
        "context_relevance": 0.85  # Relevance to the domain
    }}}}
}}}}

Consider these factors when evaluating confidence:
1. Quality and coherence of extracted content
2. Number and relevance of extracted entities
3. Validity and logic of relationships
4. Source type and reliability
5. Domain relevance and context

Provide detailed reasoning for your confidence assessment."""

CONFIDENCE_EVALUATION_HUMAN = """Evaluate confidence for this extracted knowledge:

Content: {content}

Entities: {entities}

Relationships: {relationships}

Source Type: {source_type}

Output ONLY a valid JSON object following the specified format.""" 