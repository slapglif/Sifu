"""Synthetic knowledge prompts package."""

from .knowledge_generation import (
    Pattern,
    Hypothesis,
    Relationship,
    SyntheticKnowledge,
    get_pattern_recognition_prompt,
    get_hypothesis_generation_prompt,
    get_relationship_inference_prompt,
    get_knowledge_generation_prompt
)

from .knowledge_synthesis import (
    get_knowledge_synthesis_prompt
)

from .join_decision import (
    get_join_decision_prompt
)

__all__ = [
    # Knowledge Generation
    'Pattern',
    'Hypothesis',
    'Relationship',
    'SyntheticKnowledge',
    'get_pattern_recognition_prompt',
    'get_hypothesis_generation_prompt',
    'get_relationship_inference_prompt',
    'get_knowledge_generation_prompt',
    
    # Knowledge Synthesis
    'get_knowledge_synthesis_prompt',
    
    # Join Decision
    'get_join_decision_prompt'
] 