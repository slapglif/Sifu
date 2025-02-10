"""Example generation module."""
from typing import Dict, Any, List, Optional, Literal, Set
from datetime import datetime
from pydantic import BaseModel, Field, SecretStr
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from scripts.chat_langchain import ChatLangChain
from scripts.logging_config import log_error_with_traceback, log_warning_with_context, log_info_with_context
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
   - Must be well-structured

Example valid response:
{{
  "examples": [
    {{
      "input_text": "What is the key difference between supervised and unsupervised learning in machine learning?",
      "output_text": "The key difference between supervised and unsupervised learning is that supervised learning uses labeled training data with known correct outputs, while unsupervised learning works with unlabeled data to discover patterns and relationships. For example, in supervised learning, a model might learn to classify emails as spam or not spam based on examples of pre-labeled emails. In contrast, unsupervised learning might group similar emails together without any predefined categories.",
      "metadata": {{"source": "content"}},
      "quality_score": 0.9,
      "example_type": "knowledge_recall",
      "difficulty": 0.2,
      "skills_tested": ["comprehension", "analysis"]
    }},
    {{
      "input_text": "Given a dataset with high dimensionality, explain how you would apply dimensionality reduction techniques and what factors you would consider in choosing between PCA and t-SNE.",
      "output_text": "To address high dimensionality, I would follow these steps:\\n1. First assess the characteristics of the dataset (size, feature types, relationships)\\n2. Consider computational resources and visualization needs\\n3. Compare PCA and t-SNE based on:\\n   - PCA: Linear reduction, preserves global structure, faster\\n   - t-SNE: Non-linear reduction, preserves local structure, slower\\n4. Choose based on:\\n   - If global variance patterns are important: Use PCA\\n   - If local cluster visualization is priority: Use t-SNE\\n   - If speed is critical: Use PCA\\n   - If non-linear relationships exist: Consider t-SNE",
      "metadata": {{"source": "content"}},
      "quality_score": 0.95,
      "example_type": "critical_thinking",
      "difficulty": 0.8,
      "skills_tested": ["analysis", "evaluation", "problem_solving"]
    }}
  ],
  "metadata": {{
    "generation_method": "prompt-based",
    "timestamp": "2024-02-09T22:48:31Z"
  }}
}}"""),
            ("human", "Generate training examples from this content:\n{content}")
        ])
        
    async def generate_examples(self, content: str) -> ExampleGenerationResult:
        """Generate examples from content."""
        try:
            # Generate examples using LLM
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "content": content,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Validate and filter examples
            validated_examples = []
            for example in result.examples:
                if await self._validate_example(example, content):
                    validated_examples.append(example)
                    
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
            
            return result
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating examples")
            return ExampleGenerationResult()
            
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

            # Check quality factors
            quality_factors = {
                "input_clarity": self._assess_input_clarity(example.input_text),
                "output_completeness": self._assess_output_completeness(example.output_text),
                "relevance": self._assess_relevance(example, content),
                "skills_coverage": len(example.skills_tested) / 5.0  # Normalize to 0-1
            }
            
            # Calculate overall quality score with weighted factors
            weights = {
                "input_clarity": 0.3,
                "output_completeness": 0.3,
                "relevance": 0.25,
                "skills_coverage": 0.15
            }
            quality_score = sum(score * weights[factor] for factor, score in quality_factors.items())
            
            # Update example's quality score
            example.quality_score = quality_score
            
            if quality_score < 0.65:  # Increased from 0.6
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
                
            # Check difficulty balance with adaptive thresholds
            current_count = self._difficulty_counts[category]
            total_examples = sum(self._difficulty_counts.values())
            
            # Calculate current distribution
            if total_examples > 0:
                basic_ratio = self._difficulty_counts['basic'] / total_examples
                intermediate_ratio = self._difficulty_counts['intermediate'] / total_examples
                advanced_ratio = self._difficulty_counts['advanced'] / total_examples
                
                # Define target ratios with wider tolerance
                target_ratios = {
                    'basic': (0.35, 0.45),  # 35-45% basic
                    'intermediate': (0.35, 0.45),  # 35-45% intermediate
                    'advanced': (0.15, 0.25)  # 15-25% advanced
                }
                
                # Check if current category is within acceptable range
                min_ratio, max_ratio = target_ratios[category]
                current_ratio = self._difficulty_counts[category] / total_examples
                
                # More permissive in early stages
                if total_examples < 10:
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
            
            # Check example type distribution
            type_count = self._type_counts.get(example.example_type, 0)
            max_per_type = max(3, total_examples // 4)  # Allow at most 25% of one type
            
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
        if words < 5:
            score += 0.1
        elif words < 10:
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
        if input_text[0].isupper() and input_text[-1] in '.?':  # Proper capitalization and punctuation
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
        if words < 10:
            score += 0.1
        elif words < 20:
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
        if output_text[0].isupper() and output_text[-1] in '.':  # Proper capitalization and punctuation
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
        
        score += (input_overlap * 0.15) + (output_overlap * 0.25)
        
        # Key concept scoring (0.0-0.3)
        key_concepts = self._extract_key_concepts(content)
        concept_count = sum(1 for concept in key_concepts 
                          if concept in example.input_text.lower() 
                          or concept in example.output_text.lower())
        score += min(0.3, concept_count * 0.1)
        
        # Context relevance scoring (0.0-0.3)
        if example.example_type == "knowledge_recall":
            score += 0.3 if input_overlap > 0.3 else 0.1
        elif example.example_type in ["concept_application", "analysis"]:
            score += 0.3 if output_overlap > 0.3 else 0.1
        else:  # problem_solving, critical_thinking
            score += 0.3 if (input_overlap + output_overlap) / 2 > 0.3 else 0.1
            
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