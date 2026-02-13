"""
Labeling module.

LLM-based span labeling using Claude API to classify text spans
as factual-retrieval vs. generation, with quality validation.
"""

from src.labeling.llm_labeler import LLMLabeler
from src.labeling.label_validator import LabelValidator

__all__ = [
    "LLMLabeler",
    "LabelValidator",
]