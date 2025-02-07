"""Text inspection prompts package."""

from .text_analysis import (
    TextSegment,
    Entity,
    TextAnalysis,
    get_text_analysis_prompt
)

__all__ = [
    'TextSegment',
    'Entity',
    'TextAnalysis',
    'get_text_analysis_prompt'
] 