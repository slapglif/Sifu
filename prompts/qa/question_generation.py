"""Question generation prompts."""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class Question(BaseModel):
    """Question model."""
    question: str = Field(description="The question text")
    type: Literal["knowledge_recall", "concept_application", "analysis", "problem_solving", "critical_thinking"] = Field(description="Type of question")
    difficulty: float = Field(description="Difficulty score between 0 and 1", ge=0.0, le=1.0)
    topic: str = Field(description="Topic of the question")
    expected_answer: str = Field(description="Expected answer to the question")
    skills_tested: List[str] = Field(description="Skills being tested by the question")

class QuestionList(BaseModel):
    """List of questions."""
    questions: List[Question] = Field(description="List of generated questions")

def get_question_generation_prompt() -> ChatPromptTemplate:
    """Get question generation prompt."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert at generating high-quality questions and answers for knowledge assessment.
Given a topic and context, generate diverse questions that test different aspects of understanding.
Each question must have a clear, comprehensive answer that directly addresses the question.

{format_instructions}

CRITICAL RULES:
1. Question Types (MUST use EXACTLY these types with underscores, not hyphens):
   - "knowledge_recall": Testing basic understanding and recall
   - "concept_application": Testing application of concepts
   - "analysis": Testing analysis and breakdown
   - "problem_solving": Testing practical problem solving (use underscore, NOT hyphen)
   - "critical_thinking": Testing evaluation and synthesis

   IMPORTANT: Always use underscores (_) not hyphens (-) in type names.
   Example: "problem_solving" is correct, "problem-solving" is WRONG.

2. Question Distribution:
   - Aim for balanced mix of types
   - No more than 30% of one type
   - Include at least one of each type

3. Question Quality:
   - Questions must be clear and specific
   - No yes/no questions
   - Use proper grammar and punctuation
   - Include context when needed
   - Questions should be answerable from the provided context

4. Answer Quality:
   - Answers must directly address the question
   - Include explanations and reasoning
   - Use specific examples from context
   - Be comprehensive but concise
   - Include relevant details and evidence

5. Difficulty Levels:
   - Basic (0.0-0.3): Simple recall and understanding
   - Intermediate (0.3-0.7): Application and analysis
   - Advanced (0.7-1.0): Synthesis and evaluation
   - Distribute evenly across levels

6. Skills Coverage:
   Each question should test 2-3 skills from:
   - Comprehension
   - Application
   - Analysis
   - Synthesis
   - Evaluation
   - Problem Solving
   - Critical Thinking
   - Pattern Recognition

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text.
DO NOT include ```json or ``` markers.
DO NOT include any explanatory text.
ENSURE all JSON is properly escaped and formatted."""),
        ("human", "Generate {num_questions} questions about this topic:\n{topic}\n\nContext:\n{context}")
    ]) 