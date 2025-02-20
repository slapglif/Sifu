"""Example generation module."""
from typing import Dict, Any, List, Optional, Literal, Set, cast
from datetime import datetime
from pydantic import BaseModel, Field, SecretStr
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from scripts.chat_langchain import ChatLangChain
from scripts.qa_system import QASystem, Question
from scripts.text_web_browser_fixed import SimpleTextBrowser, web_search
from scripts.logging_config import log_error_with_traceback, log_warning_with_context, log_info_with_context
from langchain_neo4j import Neo4jGraph
import os

class Example(BaseModel):
    """Training example model."""
    input_text: str = Field(description="Input text for the example")
    output_text: str = Field(description="Expected output text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    quality_score: float = Field(default=0.8, description="Quality score between 0 and 1", ge=0.0, le=1.0)
    example_type: Literal["knowledge_recall", "concept_application", "analysis", "problem_solving", "critical_thinking"] = Field(description="Type of example")
    difficulty: float = Field(default=0.5, description="Difficulty score between 0 and 1", ge=0.0, le=1.0)
    skills_tested: List[str] = Field(default_factory=list, description="Skills being tested")

class ExampleGenerationResult(BaseModel):
    """Result of example generation."""
    examples: List[Example] = Field(default_factory=list, description="Generated examples")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ExampleGenerator:
    """Generates training examples from content."""
    def __init__(self, config: Dict[str, Any]):
        """Initialize example generator."""
        self.config = config
        
        # Initialize output parser
        self.parser = PydanticOutputParser(pydantic_object=ExampleGenerationResult)
        
        # Initialize LLM with ChatLangChain
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable must be set")
            
        self.llm = ChatLangChain(
            model="gemini-1.5-flash",
            temperature=0.7,
            api_key=SecretStr(api_key),
            pydantic_schema=ExampleGenerationResult,
            format='json',
            response_format={"type": "json_object"}
        )
        
        # Initialize QA system
        graph = config.get("graph")
        if not isinstance(graph, Neo4jGraph):
            raise ValueError("Config must contain a valid Neo4jGraph instance")
            
        self.qa_system = QASystem(
            graph=graph,
            llm=self.llm
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=config.get("collection_name", "examples"),
            embedding_function=OllamaEmbeddings(model='bge-m3'),
            persist_directory=config.get("persist_directory", "./data/chroma")
        )
        
        # Initialize web browser for augmentation
        self.browser = SimpleTextBrowser()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at generating high-quality training examples.
Given some content, generate diverse training examples that capture different aspects of the knowledge.
Each example should have an input question/prompt and the expected output answer/response.
The examples should be varied in format, complexity and skills tested.

{format_instructions}

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
DO NOT wrap the JSON in markdown code blocks or any other formatting.
DO NOT include ```json or ``` markers.
DO NOT include any text before or after the JSON.
DO NOT include any explanatory text or comments.
ENSURE all JSON is properly escaped and formatted.

CRITICAL EXAMPLE TYPE RULES:
You MUST use ONLY these exact example types:
- "knowledge_recall" - For testing basic understanding and recall
- "concept_application" - For applying concepts in context
- "analysis" - For analyzing and breaking down concepts
- "problem_solving" - For solving practical problems
- "critical_thinking" - For evaluation and synthesis

IMPORTANT INSTRUCTIONS:
1. Example Type Distribution:
   - knowledge_recall: ~25% of examples
   - concept_application: ~25% of examples
   - analysis: ~20% of examples
   - problem_solving: ~15% of examples
   - critical_thinking: ~15% of examples
   
2. Difficulty Distribution:
   - Basic (0.0-0.3): 35-45% of examples
     * Simple recall questions
     * Basic understanding checks
     * Straightforward definitions
   
   - Intermediate (0.3-0.7): 35-45% of examples
     * Application questions
     * Analysis tasks
     * Compare/contrast exercises
   
   - Advanced (0.7-1.0): 15-25% of examples
     * Complex problem solving
     * Synthesis tasks
     * Critical evaluation
   
3. Required Skills Coverage:
   Each example MUST test at least 2 skills from:
   - Comprehension (understanding concepts)
   - Application (using knowledge)
   - Analysis (breaking down concepts)
   - Synthesis (combining ideas)
   - Evaluation (assessing and judging)
   - Problem Solving (practical application)
   - Critical Thinking (reasoned judgment)
   - Pattern Recognition (identifying trends)
   
4. Quality Requirements:
   Input Text:
   - Must be clear and unambiguous
   - Must use proper grammar and punctuation
   - Must be appropriately complex for difficulty level
   - Must include context when needed
   - Must use proper terminology
   
   Output Text:
   - Must be comprehensive and complete
   - Must directly answer the input
   - Must include explanations where appropriate
   - Must demonstrate understanding
   - Must use proper terminology
   - Must be well-structured"""),
            ("human", "Generate training examples from this content:\n{content}\nKnowledge Graph Context:\n{kg_context}\nSimilar Examples:\n{similar_examples}")
        ])
        
    async def generate_examples(self, content: str) -> ExampleGenerationResult:
        """Generate examples from content."""
        try:
            # Get knowledge graph context
            kg_context = await self._get_kg_context(content)
            
            # Get similar examples from vector store
            similar_examples = await self._get_similar_examples(content)
            
            # Generate base examples using LLM
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "content": content,
                "kg_context": kg_context,
                "similar_examples": similar_examples,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Generate additional QA pairs using QA system
            qa_examples = await self._generate_qa_examples(content)
            
            # Augment with web knowledge
            web_examples = await self._augment_with_web_knowledge(content)
            
            # Combine all examples
            all_examples = result.examples + qa_examples + web_examples
            
            # Validate and filter examples
            validated_examples = []
            for example in all_examples:
                if await self._validate_example(example, content):
                    validated_examples.append(example)
                    
            # Ensure minimum number of examples
            while len(validated_examples) < 100:
                # Generate more examples with different prompts/approaches
                more_examples = await self._generate_more_examples(content, len(validated_examples))
                for example in more_examples:
                    if await self._validate_example(example, content):
                        validated_examples.append(example)
                        if len(validated_examples) >= 500:  # Cap at 500
                            break
                            
            # Update result with validated examples
            result.examples = validated_examples
            result.metadata.update({
                "num_examples": len(validated_examples),
                "avg_quality": sum(ex.quality_score for ex in validated_examples) / len(validated_examples) if validated_examples else 0,
                "avg_difficulty": sum(ex.difficulty for ex in validated_examples) / len(validated_examples) if validated_examples else 0,
                "example_types": list(set(ex.example_type for ex in validated_examples)),
                "skills_covered": list(set(skill for ex in validated_examples for skill in ex.skills_tested)),
                "timestamp": datetime.now().isoformat()
            })
            
            # Save examples to vector store
            await self._save_to_vector_store(validated_examples)
            
            return result
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating examples")
            return ExampleGenerationResult()
            
    async def _get_kg_context(self, content: str) -> str:
        """Get relevant context from knowledge graph."""
        try:
            # Extract key terms
            key_terms = self._extract_key_concepts(content)
            
            # Query graph for related information
            context_parts = []
            for term in key_terms:
                results = self.qa_system.graph.query(f"""
                    MATCH (n)-[r]-(m)
                    WHERE toLower(n.name) CONTAINS toLower($term)
                    RETURN n.name as source, type(r) as relation, m.name as target
                    LIMIT 10
                """, {"term": term})
                
                for result in results:
                    context_parts.append(f"{result['source']} {result['relation']} {result['target']}")
                    
            return "\n".join(context_parts)
            
        except Exception as e:
            log_error_with_traceback(e, "Error getting KG context")
            return ""
            
    async def _get_similar_examples(self, content: str) -> List[Dict[str, Any]]:
        """Get similar examples from vector store."""
        try:
            # Get embeddings for content
            embeddings = OllamaEmbeddings(model='bge-m3').embed_documents([content])[0]
            
            # Search vector store
            results = self.vector_store.similarity_search_by_vector(
                embeddings,
                k=10
            )
            
            return [doc.metadata for doc in results if "example" in doc.metadata]
            
        except Exception as e:
            log_error_with_traceback(e, "Error getting similar examples")
            return []
            
    async def _generate_qa_examples(self, content: str) -> List[Example]:
        """Generate examples using QA system."""
        try:
            # Generate questions
            questions = await self.qa_system.generate_questions(content, num_questions=20)
            
            # Generate answers
            examples = []
            for question in questions:
                response = await self.qa_system.process_qa_chain(question.question)
                if response and response.answer:
                    example = Example(
                        input_text=question.question,
                        output_text=response.answer,
                        metadata={
                            "source": "qa_system",
                            "confidence": response.confidence,
                            "sources": response.sources
                        },
                        quality_score=response.confidence,
                        example_type=cast(
                            Literal["knowledge_recall", "concept_application", "analysis", "problem_solving", "critical_thinking"],
                            self._map_question_type(question.type)
                        ),
                        difficulty=question.difficulty,
                        skills_tested=["comprehension", "analysis"]  # Default skills
                    )
                    examples.append(example)
                    
            return examples
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating QA examples")
            return []
            
    async def _augment_with_web_knowledge(self, content: str) -> List[Example]:
        """Augment examples with web knowledge."""
        try:
            # Search web for related content with config
            config = {
                "domain_name": "medical research",  # Default domain
                "search_depth": 2,
                "max_results": 10
            }
            search_results = await web_search(content[:200], config)  # Use first 200 chars as query
            
            # Generate examples from web content
            examples = []
            for result in search_results.split("---"):
                if result.strip():
                    # Generate examples from web content
                    chain = self.prompt | self.llm | self.parser
                    result = await chain.ainvoke({
                        "content": result,
                        "kg_context": "",
                        "similar_examples": "",
                        "format_instructions": self.parser.get_format_instructions()
                    })
                    
                    if result and result.examples:
                        for example in result.examples:
                            example.metadata["source"] = "web"
                        examples.extend(result.examples)
                        
            return examples
            
        except Exception as e:
            log_error_with_traceback(e, "Error augmenting with web knowledge")
            return []
            
    async def _generate_more_examples(self, content: str, current_count: int) -> List[Example]:
        """Generate additional examples to meet minimum requirements."""
        try:
            # Try different approaches based on current count
            examples = []
            
            # Approach 1: Use different temperature
            if current_count < 200:
                temp_llm = ChatLangChain(
                    model="gemini-1.5-flash",
                    temperature=0.9,  # Higher temperature for more variety
                    api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "")),
                    pydantic_schema=ExampleGenerationResult,
                    format='json'
                )
                chain = self.prompt | temp_llm | self.parser
                result = await chain.ainvoke({
                    "content": content,
                    "kg_context": await self._get_kg_context(content),
                    "similar_examples": "",
                    "format_instructions": self.parser.get_format_instructions()
                })
                if result and result.examples:
                    examples.extend(result.examples)
                    
            # Approach 2: Focus on specific example types
            if current_count < 300:
                for example_type in ["problem_solving", "critical_thinking"]:
                    questions = await self.qa_system.generate_questions(
                        content,
                        num_questions=10
                    )
                    for question in questions:
                        response = await self.qa_system.process_qa_chain(question.question)
                        if response and response.answer:
                            example = Example(
                                input_text=question.question,
                                output_text=response.answer,
                                metadata={"source": "focused_qa"},
                                quality_score=response.confidence,
                                example_type=cast(
                                    Literal["knowledge_recall", "concept_application", "analysis", "problem_solving", "critical_thinking"],
                                    example_type
                                ),
                                difficulty=0.7,
                                skills_tested=["critical_thinking", "problem_solving"]
                            )
                            examples.append(example)
                            
            # Approach 3: Use web knowledge more aggressively
            if current_count < 400:
                web_examples = await self._augment_with_web_knowledge(content)
                examples.extend(web_examples)
                
            return examples
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating more examples")
            return []
            
    async def _save_to_vector_store(self, examples: List[Example]) -> None:
        """Save examples to vector store."""
        try:
            texts = []
            metadatas = []
            
            for example in examples:
                # Combine input and output for embedding
                text = f"Question: {example.input_text}\nAnswer: {example.output_text}"
                texts.append(text)
                
                # Store full example in metadata
                metadatas.append({
                    "example": example.model_dump(),
                    "timestamp": datetime.now().isoformat()
                })
                
            # Generate embeddings
            embeddings = OllamaEmbeddings(model='bge-m3').embed_documents(texts)
            
            # Add to vector store
            self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        except Exception as e:
            log_error_with_traceback(e, "Error saving to vector store")
            
    def _map_question_type(self, qa_type: str) -> str:
        """Map QA system question type to example type."""
        mapping = {
            "factual": "knowledge_recall",
            "conceptual": "concept_application",
            "analytical": "analysis",
            "problem": "problem_solving",
            "synthesis": "critical_thinking"
        }
        return mapping.get(qa_type, "knowledge_recall")

    async def _validate_example(self, example: Example, content: str) -> bool:
        """Validate a generated example."""
        try:
            # Check input text quality
            if not example.input_text or len(example.input_text.split()) < 3:
                log_warning_with_context("Input text too short or empty", "Example Generation")
                return False
                
            # Check output text quality
            if not example.output_text or len(example.output_text.split()) < 5:
                log_warning_with_context("Output text too short or empty", "Example Generation")
                return False
                
            # Check if example tests content knowledge
            input_relevance = any(term.lower() in example.input_text.lower() for term in content.split())
            output_relevance = any(term.lower() in example.output_text.lower() for term in content.split())
            if not (input_relevance or output_relevance):
                log_warning_with_context("Example not relevant to content", "Example Generation")
                return False

            # Check skills coverage
            if not example.skills_tested:
                log_warning_with_context("No skills specified", "Example Generation")
                return False
            
            # Ensure diverse skills
            if len(example.skills_tested) < 2:
                log_warning_with_context("Too few skills tested", "Example Generation")
                return False

            # Check quality factors with lower thresholds
            quality_factors = {
                "input_clarity": self._assess_input_clarity(example.input_text),
                "output_completeness": self._assess_output_completeness(example.output_text),
                "relevance": self._assess_relevance(example, content),
                "skills_coverage": len(example.skills_tested) / 4.0  # Normalize to 0-1, reduced from 5.0
            }
            
            # Calculate overall quality score with adjusted weights
            weights = {
                "input_clarity": 0.25,  # Reduced from 0.3
                "output_completeness": 0.25,  # Reduced from 0.3
                "relevance": 0.3,  # Increased from 0.25
                "skills_coverage": 0.2  # Increased from 0.15
            }
            quality_score = sum(score * weights[factor] for factor, score in quality_factors.items())
            
            # Update example's quality score
            example.quality_score = quality_score
            
            if quality_score < 0.55:  # Reduced from 0.65
                log_warning_with_context(f"Example quality too low: {quality_score:.2f}", "Example Generation")
                return False
                
            # Initialize difficulty counts if not exists
            if not hasattr(self, '_difficulty_counts'):
                self._difficulty_counts = {
                    'basic': 0,      # 0.0-0.3
                    'intermediate': 0,  # 0.3-0.7
                    'advanced': 0    # 0.7-1.0
                }
                
            # Categorize difficulty
            if example.difficulty <= 0.3:
                category = 'basic'
            elif example.difficulty <= 0.7:
                category = 'intermediate'
            else:
                category = 'advanced'
                
            # Check difficulty balance with more permissive thresholds
            current_count = self._difficulty_counts[category]
            total_examples = sum(self._difficulty_counts.values())
            
            # Calculate current distribution
            if total_examples > 0:
                basic_ratio = self._difficulty_counts['basic'] / total_examples
                intermediate_ratio = self._difficulty_counts['intermediate'] / total_examples
                advanced_ratio = self._difficulty_counts['advanced'] / total_examples
                
                # Define target ratios with wider tolerance
                target_ratios = {
                    'basic': (0.30, 0.50),  # 30-50% basic (widened from 35-45%)
                    'intermediate': (0.30, 0.50),  # 30-50% intermediate (widened from 35-45%)
                    'advanced': (0.10, 0.30)  # 10-30% advanced (widened from 15-25%)
                }
                
                # Check if current category is within acceptable range
                min_ratio, max_ratio = target_ratios[category]
                current_ratio = self._difficulty_counts[category] / total_examples
                
                # More permissive in early stages
                if total_examples < 20:  # Increased from 10
                    # Allow any distribution initially
                    pass
                elif current_ratio > max_ratio:
                    log_warning_with_context(f"Too many {category} difficulty examples", "Example Generation")
                    return False
                elif current_ratio < min_ratio:
                    # Encourage this category
                    pass
                
            # Update difficulty counts
            self._difficulty_counts[category] += 1
            
            # Track example type distribution
            if not hasattr(self, '_type_counts'):
                self._type_counts = {
                    'knowledge_recall': 0,
                    'concept_application': 0,
                    'analysis': 0,
                    'problem_solving': 0,
                    'critical_thinking': 0
                }
            
            # Check example type distribution with more permissive limits
            type_count = self._type_counts.get(example.example_type, 0)
            max_per_type = max(5, total_examples // 3)  # Allow up to 33% of one type (increased from 25%)
            
            if type_count >= max_per_type:
                log_warning_with_context(f"Too many examples of type {example.example_type}", "Example Generation")
                return False
                
            self._type_counts[example.example_type] = type_count + 1
            return True
            
        except Exception as e:
            log_error_with_traceback(e, "Error validating example")
            return False

    def _assess_input_clarity(self, input_text: str) -> float:
        """Assess clarity of input text."""
        score = 0.0
        
        # Length scoring (0.0-0.3)
        words = len(input_text.split())
        if words < 4:  # Reduced from 5
            score += 0.1
        elif words < 8:  # Reduced from 10
            score += 0.2
        else:
            score += 0.3
            
        # Structure scoring (0.0-0.4)
        if '?' in input_text:
            score += 0.1
        if any(input_text.lower().startswith(w) for w in ['what', 'why', 'how', 'when', 'where', 'who', 'which']):
            score += 0.1
        if len(input_text.split(',')) > 1 or len(input_text.split('and')) > 1:
            score += 0.1
        if any(phrase in input_text.lower() for phrase in ['explain', 'describe', 'compare', 'analyze', 'evaluate']):
            score += 0.1
            
        # Quality scoring (0.0-0.3)
        if input_text[0].isupper():  # Removed period/question mark requirement
            score += 0.1
        if not any(char in input_text for char in '@#$%^&*'):  # No special characters
            score += 0.1
        if len(input_text) >= len(input_text.strip()):  # No extra whitespace
            score += 0.1
            
        return min(1.0, score)

    def _assess_output_completeness(self, output_text: str) -> float:
        """Assess completeness of output text."""
        score = 0.0
        
        # Length scoring (0.0-0.3)
        words = len(output_text.split())
        if words < 8:  # Reduced from 10
            score += 0.1
        elif words < 15:  # Reduced from 20
            score += 0.2
        else:
            score += 0.3
            
        # Structure scoring (0.0-0.4)
        sentences = len(output_text.split('.'))
        if sentences > 1:
            score += 0.1
        if any(phrase in output_text.lower() for phrase in ['because', 'therefore', 'thus', 'hence']):
            score += 0.1
        if any(phrase in output_text.lower() for phrase in ['for example', 'such as', 'specifically']):
            score += 0.1
        if any(phrase in output_text.lower() for phrase in ['first', 'second', 'finally', 'additionally']):
            score += 0.1
            
        # Quality scoring (0.0-0.3)
        if output_text[0].isupper():  # Removed period requirement
            score += 0.1
        if not any(char in output_text for char in '@#$%^&*'):  # No special characters
            score += 0.1
        if len(output_text) >= len(output_text.strip()):  # No extra whitespace
            score += 0.1
            
        return min(1.0, score)

    def _assess_relevance(self, example: Example, content: str) -> float:
        """Assess relevance to content."""
        score = 0.0
        
        # Content overlap scoring (0.0-0.4)
        content_terms = set(content.lower().split())
        input_terms = set(example.input_text.lower().split())
        output_terms = set(example.output_text.lower().split())
        
        input_overlap = len(input_terms.intersection(content_terms)) / len(input_terms) if input_terms else 0
        output_overlap = len(output_terms.intersection(content_terms)) / len(output_terms) if output_terms else 0
        
        score += (input_overlap * 0.2) + (output_overlap * 0.2)  # More balanced between input/output
        
        # Key concept scoring (0.0-0.3)
        key_concepts = self._extract_key_concepts(content)
        concept_count = sum(1 for concept in key_concepts 
                          if concept in example.input_text.lower() 
                          or concept in example.output_text.lower())
        score += min(0.3, concept_count * 0.15)  # Increased from 0.1
        
        # Context relevance scoring (0.0-0.3)
        if example.example_type == "knowledge_recall":
            score += 0.3 if input_overlap > 0.2 else 0.15  # Reduced threshold from 0.3
        elif example.example_type in ["concept_application", "analysis"]:
            score += 0.3 if output_overlap > 0.2 else 0.15  # Reduced threshold from 0.3
        else:  # problem_solving, critical_thinking
            score += 0.3 if (input_overlap + output_overlap) / 2 > 0.2 else 0.15  # Reduced threshold from 0.3
            
        return min(1.0, score)

    def _extract_key_concepts(self, content: str) -> Set[str]:
        """Extract key concepts from content."""
        # Simple extraction based on capitalized terms and common ML concepts
        concepts = set()
        
        # Add capitalized terms
        words = content.split()
        concepts.update(word.lower() for word in words if word[0].isupper())
        
        # Add common ML concepts
        ml_concepts = {
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'training', 'model', 'algorithm', 'data', 'prediction', 'classification',
            'regression', 'accuracy', 'precision', 'recall', 'validation'
        }
        concepts.update(concept for concept in ml_concepts if concept in content.lower())
        
        return concepts 