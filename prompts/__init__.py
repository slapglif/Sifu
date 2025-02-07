"""Prompts package for the research agent system."""

from .knowledge_acquisition import (
    Relationship,
    SourceMetadata,
    ExtractedKnowledge,
    get_knowledge_extraction_prompt,
    ConfidenceFactors,
    ConfidenceEvaluation,
    get_confidence_evaluation_prompt
)

from .qa import (
    Question,
    get_question_generation_prompt,
    Answer,
    get_answer_generation_prompt,
    KnowledgeGap,
    get_knowledge_gap_prompt
)

from .synthetic_knowledge import (
    Pattern,
    Hypothesis,
    Relationship,
    SyntheticKnowledge,
    get_pattern_recognition_prompt,
    get_hypothesis_generation_prompt,
    get_relationship_inference_prompt,
    get_knowledge_synthesis_prompt
)

from .visual_qa import (
    Region,
    VisualElement,
    VisualAttributes,
    SceneAnalysis,
    VisualAnswer,
    get_element_detection_prompt,
    get_scene_analysis_prompt,
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
    'Relationship',
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