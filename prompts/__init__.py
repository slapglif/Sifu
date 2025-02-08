"""Prompts package for the research agent system."""

from .knowledge_acquisition.extraction import (
    Relationship,
    SourceMetadata,
    ExtractedKnowledge,
    get_knowledge_extraction_prompt
)

from .knowledge_acquisition.confidence_evaluation import (
    ConfidenceFactors,
    ConfidenceEvaluation,
    get_confidence_evaluation_prompt
)

from .qa.question_generation import (
    Question,
    get_question_generation_prompt
)

from .qa.answer_generation import (
    Answer,
    get_answer_generation_prompt
)

from .qa.knowledge_gap import (
    KnowledgeGap,
    get_knowledge_gap_prompt
)

from .synthetic_knowledge.pattern_recognition import (
    Pattern,
    get_pattern_recognition_prompt
)

from .synthetic_knowledge.hypothesis_generation import (
    Hypothesis,
    get_hypothesis_generation_prompt
)

from .synthetic_knowledge.relationship_inference import (
    get_relationship_inference_prompt
)

from .synthetic_knowledge.knowledge_synthesis import (
    SyntheticKnowledge,
    get_knowledge_synthesis_prompt
)

from .visual_qa.element_detection import (
    Region,
    VisualElement,
    get_element_detection_prompt
)

from .visual_qa.scene_analysis import (
    VisualAttributes,
    SceneAnalysis,
    get_scene_analysis_prompt
)

from .visual_qa.visual_qa_prompts import (
    VisualAnswer,
    get_visual_qa_prompt
)

__all__ = [
    # Knowledge Acquisition
    'Relationship',
    'SourceMetadata', 
    'ExtractedKnowledge',
    'get_knowledge_extraction_prompt',
    'ConfidenceFactors',
    'ConfidenceEvaluation', 
    'get_confidence_evaluation_prompt',

    # QA
    'Question',
    'get_question_generation_prompt',
    'Answer',
    'get_answer_generation_prompt',
    'KnowledgeGap',
    'get_knowledge_gap_prompt',

    # Synthetic Knowledge
    'Pattern',
    'Hypothesis',
    'SyntheticKnowledge',
    'get_pattern_recognition_prompt',
    'get_hypothesis_generation_prompt',
    'get_relationship_inference_prompt',
    'get_knowledge_synthesis_prompt',

    # Visual QA
    'Region',
    'VisualElement',
    'VisualAttributes',
    'SceneAnalysis',
    'VisualAnswer',
    'get_element_detection_prompt',
    'get_scene_analysis_prompt',
    'get_visual_qa_prompt'
] 