"""Question answering prompts package."""

from .qa_prompt import (
    QAResponse,
    get_qa_prompt
)

from .question_generation import (
    Question,
    QuestionList,
    get_question_generation_prompt
)

from .answer_generation import (
    Answer,
    get_answer_generation_prompt
)

from .knowledge_gap import (
    KnowledgeGap,
    get_knowledge_gap_prompt
)

__all__ = [
    'QAResponse',
    'get_qa_prompt',
    'Question',
    'get_question_generation_prompt',
    'Answer',
    'get_answer_generation_prompt',
    'KnowledgeGap',
    'get_knowledge_gap_prompt'
]