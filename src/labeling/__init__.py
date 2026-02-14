"""
Labeling module.

LLM-based span labeling using Claude API to classify text spans
as factual-retrieval vs. generation, with quality validation.
"""

from src.labeling.llm_labeler import CostTracker, LLMLabeler, SpanLabel
from src.labeling.label_validator import LabelValidator

# Alias per the Phase 5 spec naming convention
RouterLabelGenerator = LLMLabeler

__all__ = [
    "CostTracker",
    "LLMLabeler",
    "LabelValidator",
    "RouterLabelGenerator",
    "SpanLabel",
]